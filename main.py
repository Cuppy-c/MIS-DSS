import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import openai
from datetime import datetime
import warnings
from matplotlib.font_manager import FontProperties

font_path = "NotoSansCJK-Regular.ttf"  # 或使用绝对路径
custom_font = FontProperties(fname=font_path)

def set_chinese_font():
    """设置中文字体"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        return True
    except:
        return False


set_chinese_font()

warnings.filterwarnings('ignore')

api_key = "sk-ObXbMdavg61VYQgn494c51E327154f2bBfAf6a8fC7D1BeCa"
api_base = "https://maas-api.cn-huabei-1.xf-yun.com/v1"
MODEL_ID = "xop3qwen1b7"


def initialize_openai_client():
    """初始化OpenAI客户端，包含详细的错误处理"""
    try:
        if not api_key:
            st.error("❌ 请先配置有效的API密钥")
            return None, "API密钥未配置"

        client = openai.OpenAI(
            api_key=api_key,
            base_url=api_base
        )

        try:
            test_response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": "测试"}],
                max_tokens=5
            )
            st.success("✅ API客户端初始化成功并连接正常")
        except Exception as test_error:
            st.warning(f"⚠️ 客户端已初始化，但测试请求失败: {str(test_error)}")
        return client, "初始化成功"

    except openai.AuthenticationError:
        error_msg = "❌ API密钥认证失败，请检查密钥是否正确"
        st.error(error_msg)
        return None, error_msg
    except openai.APIConnectionError:
        error_msg = "❌ 网络连接失败，请检查网络连接和API基础URL"
        st.error(error_msg)
        return None, error_msg
    except openai.APIError as e:
        error_msg = f"❌ API错误: {str(e)}"
        st.error(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"❌ 初始化失败: {str(e)}"
        st.error(error_msg)
        return None, error_msg


client, init_status = initialize_openai_client()
if client:
    st.session_state.client = client
    st.session_state.model_id = MODEL_ID
    st.session_state.api_available = True
else:
    st.session_state.api_available = False
    st.session_state.init_status = init_status

st.sidebar.title("导航")
page = st.sidebar.radio("选择功能",
                        ["数据上传", "模型训练", "预测分析", "AI建议"])

with st.sidebar:
    st.markdown("---")
    st.subheader("API状态")

    if st.session_state.api_available:
        st.success("✅ AI服务可用")
        st.info(f"模型: {MODEL_ID}")
    else:
        st.error("❌ AI服务不可用")
        if 'init_status' in st.session_state:
            st.error(f"错误: {st.session_state.init_status}")


def load_data(uploaded_file):
    """加载上传的数据文件"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("不支持的文件格式，请上传CSV或Excel文件")
            return None
        return df
    except Exception as e:
        st.error(f"加载文件时出错: {str(e)}")
        return None


def preprocess_data(df, target_column):
    """更稳健的预处理函数"""
    df = df.copy()

    date_columns = []
    for col in df.columns:
        if ('date' in col.lower() or 'time' in col.lower() or
                '日期' in col or '时间' in col or
                pd.api.types.is_datetime64_any_dtype(df[col])):
            date_columns.append(col)

    for date_col in date_columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df[f'{date_col}_Year'] = df[date_col].dt.year
            df[f'{date_col}_Month'] = df[date_col].dt.month
            df[f'{date_col}_Day'] = df[date_col].dt.day
            df[f'{date_col}_DayOfWeek'] = df[date_col].dt.dayofweek

            df.drop(date_col, axis=1, inplace=True)
        except Exception as e:
            st.warning(f"日期列 {date_col} 处理失败: {e}")

    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())

    if target_column and target_column in df.columns and df[target_column].dtype == 'object':
        try:
            le_target = LabelEncoder()
            df[target_column] = le_target.fit_transform(df[target_column])
            st.session_state.label_encoder = le_target
            st.info(f"目标变量 '{target_column}' 已进行标签编码")
        except Exception as e:
            st.warning(f"目标变量编码失败: {e}")

    categorical_cols = df.select_dtypes(include=['object']).columns

    if target_column and target_column in categorical_cols:
        categorical_cols = categorical_cols.drop(target_column)

    label_encoders = {}
    for col in categorical_cols:
        try:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        except Exception as e:
            st.warning(f"分类列 {col} 编码失败: {e}")
            df.drop(col, axis=1, inplace=True)

    st.session_state.label_encoders = label_encoders

    return df


def determine_problem_type(df, target_column):
    """确定问题是分类还是回归"""
    if target_column not in df.columns:
        return "regression"

    unique_values = len(df[target_column].unique())
    if unique_values <= 10 or df[target_column].dtype == 'object':
        return "classification"
    else:
        return "regression"


def save_model(model, model_type):
    """保存模型到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_{model_type}_{timestamp}.pkl"
    joblib.dump(model, filename)
    return filename


def load_model(filename):
    """从文件加载模型"""
    return joblib.load(filename)


def generate_llm_advice(prompt):
    """生成AI建议"""
    try:
        if not st.session_state.api_available:
            return "AI服务当前不可用，请检查API配置"

        client = st.session_state.client
        model_id = st.session_state.model_id

        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "你是一个专业的决策支持助手。根据提供的信息，给出清晰、可行的建议。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = f"生成AI建议时出错: {str(e)}"
        st.error(error_msg)
        return f"无法获取AI建议: {error_msg}"


if page == "数据上传":
    st.title("数据上传")

    uploaded_file = st.file_uploader("上传数据文件 (CSV 或 Excel)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.success("数据加载成功!")

            st.subheader("数据摘要")
            st.write("前5行数据:")
            st.dataframe(df.head())

            st.subheader("描述性统计")
            st.write(df.describe())

            st.subheader("变量分析选择")
            target_col = st.selectbox("选择目标变量", df.columns)
            st.session_state.target_col = target_col

            problem_type = determine_problem_type(df, target_col)
            st.session_state.problem_type = problem_type
            st.write(f"检测到的问题类型: {'分类' if problem_type == 'classification' else '回归'}")

            st.subheader("数据可视化")

            plot_type = st.selectbox("选择图表类型",
                                     ["直方图", "箱线图", "散点图", "相关矩阵"])

            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

            if plot_type == "直方图" and numeric_cols:
                column = st.selectbox("选择列", numeric_cols)
                fig, ax = plt.subplots()
                sns.histplot(df[column], kde=True, ax=ax)
                st.pyplot(fig)

            elif plot_type == "箱线图" and numeric_cols:
                column = st.selectbox("选择列", numeric_cols)
                fig, ax = plt.subplots()
                sns.boxplot(y=df[column], ax=ax)
                st.pyplot(fig)

            elif plot_type == "散点图" and len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_axis = st.selectbox("X轴", numeric_cols, index=0)
                with col2:
                    y_axis = st.selectbox("Y轴", numeric_cols, index=1)

                fig, ax = plt.subplots()
                sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=target_col, ax=ax)
                st.pyplot(fig)

            elif plot_type == "相关矩阵" and numeric_cols:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", ax=ax)
                st.pyplot(fig)

            try:
                df_processed = preprocess_data(df, target_col)
                st.session_state.df = df_processed
                st.success("数据预处理完成!")

                with st.expander("查看预处理后的数据"):
                    st.dataframe(df_processed.head())
            except Exception as e:
                st.error(f"数据预处理失败: {str(e)}")

elif page == "模型训练":
    st.title("模型训练")

    if 'df' not in st.session_state or 'target_col' not in st.session_state:
        st.warning("请先上传数据并在'数据上传'页面设置目标变量")
    else:
        df = st.session_state.df
        target_col = st.session_state.target_col
        problem_type = st.session_state.problem_type

        st.write(f"当前问题类型: {'分类' if problem_type == 'classification' else '回归'}")

        if target_col not in df.columns:
            st.error(f"目标列 '{target_col}' 不在处理后的数据中。请返回数据上传页面重新选择目标变量。")
        else:
            st.subheader("模型配置")
            col1, col2 = st.columns(2)

            with col1:
                test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
                n_estimators = st.slider("树的数量", 10, 200, 100, 10)

            with col2:
                max_depth = st.slider("最大深度", 2, 20, 10, 1)
                random_state = st.number_input("随机种子", 0, 1000, 42)

            X = df.drop(columns=[target_col])
            y = df[target_col]

            feature_order = X.columns.tolist()
            st.session_state.feature_order = feature_order

            if X.empty:
                st.error("特征数据为空，请检查数据预处理结果")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state,
                    stratify=y if problem_type == "classification" else None
                )

                if st.button("训练模型"):

                    try:
                        if problem_type == "classification":
                            model = RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                random_state=random_state
                            )
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            metrics = {"准确率": accuracy}

                        else:
                            model = RandomForestRegressor(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                random_state=random_state
                            )
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            r2 = model.score(X_test, y_test)
                            metrics = {"均方误差": mse, "均方根误差": rmse, "R²分数": r2}

                        st.session_state.trained_model = model
                        st.session_state.model_type = problem_type

                        feature_importance_df = pd.DataFrame({
                            "特征": feature_order,
                            "重要性": model.feature_importances_
                        }).sort_values("重要性", ascending=False)

                        st.session_state.feature_importances = feature_importance_df
                        st.session_state.metrics = metrics
                        st.session_state.X_test = X_test
                        st.session_state.y_test = y_test
                        st.session_state.y_pred = y_pred

                        st.success("模型训练完成!")

                        model_file = save_model(model, problem_type)
                        st.session_state.model_file = model_file
                        st.write(f"模型已保存到: {model_file}")

                    except Exception as e:
                        st.error(f"模型训练失败: {str(e)}")

            if 'metrics' in st.session_state and st.session_state.metrics is not None:
                st.subheader("模型性能")

                if problem_type == "classification":
                    st.write(f"测试集准确率: {st.session_state.metrics['准确率']:.4f}")

                    cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)

                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                                xticklabels=np.unique(y),
                                yticklabels=np.unique(y))
                    ax.set_xlabel('预测标签')
                    ax.set_ylabel('真实标签')
                    st.pyplot(fig)

                else:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("均方误差(MSE)", f"{st.session_state.metrics['均方误差']:.4f}")
                    with col2:
                        st.metric("均方根误差(RMSE)", f"{st.session_state.metrics['均方根误差']:.4f}")
                    with col3:
                        st.metric("R²分数", f"{st.session_state.metrics['R²分数']:.4f}")

                    fig, ax = plt.subplots()
                    ax.scatter(st.session_state.y_test, st.session_state.y_pred, alpha=0.6)
                    ax.plot([st.session_state.y_test.min(), st.session_state.y_test.max()],
                            [st.session_state.y_test.min(), st.session_state.y_test.max()],
                            'k--', lw=2)
                    ax.set_xlabel("真实值")
                    ax.set_ylabel('预测值')
                    ax.set_title('真实值 vs 预测值')
                    st.pyplot(fig)

                st.subheader("特征重要性")
                st.dataframe(st.session_state.feature_importances)

                fig, ax = plt.subplots(figsize=(10, 6))
                top_features = st.session_state.feature_importances.head(10)
                sns.barplot(x="重要性", y="特征", data=top_features, ax=ax)
                ax.set_title('Top 10 特征重要性')
                st.pyplot(fig)

elif page == "预测分析":
    st.title("预测分析")

    if 'trained_model' not in st.session_state or st.session_state.trained_model is None:
        st.warning("请先训练模型")
    else:
        st.write("使用训练好的模型进行预测")

        model = st.session_state.trained_model
        problem_type = st.session_state.model_type

        if 'feature_order' in st.session_state:
            feature_names = st.session_state.feature_order
        else:
            feature_names = st.session_state.feature_importances['特征'].tolist()

        prediction_mode = st.radio("预测方式",
                                   ["单样本预测", "批量预测（上传文件）"])

        if prediction_mode == "单样本预测":
            st.subheader("输入预测特征")
            st.info(f"请按照以下顺序输入特征: {', '.join(feature_names)}")

            input_data = {}
            cols = st.columns(2)
            for i, feature in enumerate(feature_names):
                if i % 2 == 0:
                    container = cols[0]
                else:
                    container = cols[1]

                if 'df' in st.session_state and feature in st.session_state.df.columns:
                    feature_data = st.session_state.df[feature]
                    min_val = float(feature_data.min())
                    max_val = float(feature_data.max())
                    mean_val = float(feature_data.mean())

                    input_data[feature] = container.number_input(
                        f"{feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val
                    )
                else:
                    input_data[feature] = container.number_input(f"{feature}", value=0.0)

            if st.button("预测"):
                try:
                    input_df = pd.DataFrame([input_data])

                    input_df = input_df[feature_names]

                    st.write("输入数据特征顺序验证:")
                    st.write(list(input_df.columns))

                    if problem_type == "classification":
                        prediction = model.predict(input_df)
                        proba = model.predict_proba(input_df)

                        if 'label_encoder' in st.session_state:
                            original_prediction = st.session_state.label_encoder.inverse_transform(prediction)
                            st.success(f"预测类别: {original_prediction[0]}")
                        else:
                            st.success(f"预测类别: {prediction[0]}")

                        fig, ax = plt.subplots()
                        if 'label_encoder' in st.session_state:
                            class_names = st.session_state.label_encoder.classes_
                        else:
                            class_names = model.classes_

                        ax.bar(range(len(proba[0])), proba[0])
                        ax.set_xlabel('类别')
                        ax.set_ylabel('概率')
                        ax.set_xticks(range(len(class_names)))
                        ax.set_xticklabels(class_names, rotation=45)
                        ax.set_title('各类别预测概率')
                        st.pyplot(fig)

                    else:
                        prediction = model.predict(input_df)
                        st.success(f"预测值: {prediction[0]:.4f}")

                except Exception as e:
                    st.error(f"预测失败: {str(e)}")
                    st.info("请确保输入的特征数据格式正确，且特征顺序与训练时一致")

        else:
            st.subheader("批量预测")
            st.info(f"上传的文件必须包含以下特征（顺序不重要）: {', '.join(feature_names)}")

            uploaded_file = st.file_uploader("上传包含特征的数据文件 (CSV或Excel)",
                                             type=["csv", "xlsx"])

            if uploaded_file is not None:
                df = load_data(uploaded_file)
                if df is not None:
                    missing_features = set(feature_names) - set(df.columns)

                    if missing_features:
                        st.error(f"上传的文件缺少以下特征: {', '.join(missing_features)}")
                    else:
                        df_processed = preprocess_data(df, None)

                        df_predict = df_processed[feature_names]

                        st.write("预测数据特征顺序:")
                        st.write(list(df_predict.columns))

                        if st.button("执行批量预测"):
                            try:
                                predictions = model.predict(df_predict)

                                if problem_type == "classification":
                                    if 'label_encoder' in st.session_state:
                                        original_predictions = st.session_state.label_encoder.inverse_transform(predictions)
                                        df['预测类别'] = original_predictions
                                    else:
                                        df['预测类别'] = predictions
                                else:
                                    df['预测值'] = predictions

                                st.success("预测完成!")

                                st.write(df)

                                csv = df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    "下载预测结果",
                                    csv,
                                    "predictions.csv",
                                    "text/csv",
                                    key='download-csv'
                                )
                            except Exception as e:
                                st.error(f"批量预测失败: {str(e)}")

elif page == "AI建议":
    st.title("AI建议")

    if 'trained_model' not in st.session_state or st.session_state.trained_model is None:
        st.warning("请先训练模型并生成一些预测结果")
    else:
        st.write("基于模型预测结果获取AI建议")

        if not st.session_state.api_available:
            st.error("❌ AI服务当前不可用")
        else:
            st.success("✅ AI服务已就绪")

        model = st.session_state.trained_model
        problem_type = st.session_state.model_type
        feature_importances = st.session_state.feature_importances

        scenario = st.text_area("描述你的决策场景或预测目标:",
                                "我需要基于模型预测结果做出决策...")

        st.subheader("关键指标")
        col1, col2 = st.columns(2)

        with col1:
            main_metric = st.selectbox("主要考虑的指标",
                                       ["准确性" if problem_type == "classification" else "误差"])

        with col2:
            if problem_type == "classification":
                include_cm = st.checkbox("包含混淆矩阵信息")
            else:
                include_pred_vs_actual = st.checkbox("包含预测与实际值对比")

        prompt = f"""
        你是一个决策支持专家。基于以下信息提供专业的决策建议:

        1. 问题类型: {'分类' if problem_type == 'classification' else '回归'}
        2. 用户场景描述: {scenario}
        3. 关键模型指标: {st.session_state.metrics}
        4. 最重要的5个特征及其重要性:
        {feature_importances.head(5).to_string(index=False)}

        请提供具体的、可操作的决策建议，考虑模型的优缺点，以及如何基于预测结果优化决策过程。
        提供3-5条清晰的建议，并解释原因。
        """

        if st.button("获取AI建议"):
            with st.spinner("AI正在生成建议..."):
                advice = generate_llm_advice(prompt)
                st.subheader("AI决策建议")
                st.write(advice)