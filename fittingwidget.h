#ifndef FITTINGWIDGET_H
#define FITTINGWIDGET_H

#include <QWidget>
#include <QFutureWatcher>
#include <QApplication>
#include <QMouseEvent>
#include <QDialog>
#include <QFormLayout>
#include <QCheckBox>
#include <QLineEdit>
#include <QSpinBox>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QTableWidget>
#include <QComboBox>

// 引入项目通用模块
#include "modelmanager.h"
#include "mousezoom.h"      // 替换原有的 EnhancedCustomPlot
#include "chartsetting1.h"  // 替换原有的 ChartSettingsDialog

namespace Ui {
class FittingWidget;
}

// ===========================================================================
// FitParameter: 拟合参数结构体
// ===========================================================================
/**
 * @brief 用于存储单个拟合参数的所有属性
 * 包含参数的名称、显示符号、数值、边界条件以及是否参与拟合的标志。
 */
struct FitParameter {
    QString name;        ///< 英文内部键名 (如 "kf", "phi")，用于算法逻辑索引
    QString displayName; ///< 中文显示名 (如 "渗透率", "孔隙度")，用于界面显示
    QString symbol;      ///< 符号 (如 "k", "φ")，支持 HTML 格式
    double value;        ///< 参数当前的数值
    double min;          ///< 允许的最小值（边界约束）
    double max;          ///< 允许的最大值（边界约束）
    bool isFit;          ///< 勾选状态，true 表示该参数是变量，参与自动拟合；false 表示为固定常量
    QString unit;        ///< 单位 (如 "mD", "m³/d")
};

// ===========================================================================
// FittingWidget: 拟合主界面类
// ===========================================================================
/**
 * @brief 拟合功能的主窗口部件
 * 负责参数管理、数据加载、图表绘制、拟合算法调用（Levenberg-Marquardt）以及数据导出。
 */
class FittingWidget : public QWidget
{
    Q_OBJECT

public:
    explicit FittingWidget(QWidget *parent = nullptr);
    ~FittingWidget();

    /**
     * @brief 设置模型管理器
     * 用于获取不同模型的默认参数以及计算理论曲线。
     * @param m ModelManager 指针
     */
    void setModelManager(ModelManager* m);

    /**
     * @brief 设置实测数据，并更新图表
     * 将外部导入的实测数据加载到拟合模块中。
     * @param t 时间数组 (h)
     * @param p 压力数组 (MPa)
     * @param d 导数数组 (MPa)
     */
    void setObservedData(const QVector<double>& t, const QVector<double>& p, const QVector<double>& d);

signals:
    /**
     * @brief 迭代更新信号
     * 由后台拟合线程发出，通知 UI 刷新当前的拟合曲线和误差值。
     * @param error 当前均方误差 (MSE)
     * @param currentParams 当前尝试的参数集
     * @param t 时间序列
     * @param p 计算出的压力序列
     * @param d 计算出的导数序列
     */
    void sigIterationUpdated(double error, const QMap<QString,double>& currentParams,
                             const QVector<double>& t, const QVector<double>& p, const QVector<double>& d);

    /**
     * @brief 进度条更新信号
     * @param percent 完成百分比 (0-100)
     */
    void sigProgress(int percent);

    /**
     * @brief 拟合完成信号
     * @param type 模型类型
     * @param finalParams 最终拟合得到的参数
     */
    void fittingCompleted(ModelManager::ModelType type, const QMap<QString,double>& finalParams);

private slots:
    // --- 界面按钮响应槽 ---
    void on_btnLoadData_clicked();       ///< 点击"加载数据"：打开文件并解析
    void on_btnRunFit_clicked();         ///< 点击"自动拟合"：启动后台优化线程
    void on_btnStop_clicked();           ///< 点击"停止"：中断拟合过程
    void on_btnResetParams_clicked();    ///< 点击"重置参数"：恢复默认典型值
    void on_btnImportModel_clicked();    ///< 点击"导入模型"：根据表格参数绘制一次理论曲线
    void on_btnResetView_clicked();      ///< 点击"重置视图"：恢复坐标轴默认范围
    void on_btnExportData_clicked();     ///< 点击"导出数据"：保存参数到 CSV/TXT
    void on_btnExportChart_clicked();    ///< 点击"导出图表"：保存截图
    void on_btnChartSettings_clicked();  ///< 点击"图表设置"：弹出配置对话框 (ChartSetting1)

    void on_comboModelSelect_currentIndexChanged(int index); ///< 模型下拉框切换：重置参数列表

    // --- 逻辑处理槽 ---
    /**
     * @brief 接收后台线程传来的单次迭代结果
     * 在主线程更新 UI（图表和表格数值）。
     */
    void onIterationUpdate(double err, const QMap<QString,double>& p,
                           const QVector<double>& t, const QVector<double>& p_curve, const QVector<double>& d_curve);

    /**
     * @brief 拟合线程结束后的收尾工作
     * 恢复按钮状态，提示完成。
     */
    void onFitFinished();

private:
    Ui::FittingWidget *ui;
    ModelManager* m_modelManager;

    // 使用通用的增强型绘图控件
    MouseZoom* m_plot;
    QCPTextElement* m_plotTitle; ///< 图表标题对象指针

    bool m_isFitting;     ///< 状态位：是否正在运行拟合
    bool m_stopRequested; ///< 状态位：是否已点击停止按钮

    // 数据存储
    QVector<double> m_obsTime;       ///< 实测时间
    QVector<double> m_obsPressure;   ///< 实测压力
    QVector<double> m_obsDerivative; ///< 实测导数

    // 参数列表
    QList<FitParameter> m_parameters;

    // 用于管理后台线程的观察者
    QFutureWatcher<void> m_watcher;

    // --- 内部辅助函数 ---
    void setupPlot();          ///< 初始化图表的样式（坐标轴、图层、标题等）
    void initModelCombo();     ///< 初始化模型选择下拉框内容
    void loadParamsToTable();  ///< 将内存中的 m_parameters 渲染到 UI 表格控件中
    void updateParamsFromTable(); ///< 从 UI 表格读取用户修改的值回写到 m_parameters

    void updateModelCurve();   ///< 触发一次理论曲线计算并绘图

    /**
     * @brief 获取参数的显示属性（支持 Unicode 上下标）
     * @param key 参数键名
     * @param outName 输出：中文名称
     * @param outSymbol 输出：用于 UI 显示的 HTML 符号
     * @param outUnicodeSymbol 输出：用于文本导出的 Unicode 符号
     * @param outUnit 输出：带格式的单位字符串
     */
    void getParamDisplayInfo(const QString& key, QString& outName, QString& outSymbol, QString& outUnicodeSymbol, QString& outUnit);

    QStringList getParamOrder(ModelManager::ModelType type); ///< 获取特定模型的参数排列顺序
    QStringList parseLine(const QString& line); ///< 解析数据文件的每一行（处理逗号或空白分隔）

    /**
     * @brief 核心绘图函数
     * @param t 时间
     * @param p 压力
     * @param d 导数
     * @param isModel true绘制理论曲线(线)，false绘制实测数据(点)
     */
    void plotCurves(const QVector<double>& t, const QVector<double>& p, const QVector<double>& d, bool isModel);

    // --- 拟合算法相关 (Levenberg-Marquardt) ---
    /**
     * @brief 启动优化任务的入口函数
     */
    void runOptimizationTask(ModelManager::ModelType modelType, QList<FitParameter> fitParams, double weight);

    /**
     * @brief 执行 Levenberg-Marquardt 算法的主循环
     */
    void runLevenbergMarquardtOptimization(ModelManager::ModelType modelType, QList<FitParameter> params, double weight);

    /**
     * @brief 计算残差向量
     * 残差 = (log(实测) - log(理论)) * 权重
     */
    QVector<double> calculateResiduals(const QMap<QString,double>& params, ModelManager::ModelType modelType, double weight);

    /**
     * @brief 计算残差平方和 (SSE)
     */
    double calculateSumSquaredError(const QVector<double>& residuals);

    /**
     * @brief 计算雅可比矩阵 (Jacobian)
     * 使用有限差分法近似求导。
     */
    QVector<QVector<double>> computeJacobian(const QMap<QString,double>& params,
                                             const QVector<double>& baseResiduals,
                                             const QVector<int>& fitIndices,
                                             ModelManager::ModelType modelType,
                                             const QList<FitParameter>& currentFitParams,
                                             double weight);

    /**
     * @brief 求解线性方程组 Ax = b
     * 用于计算 LM 算法的更新步长。
     */
    QVector<double> solveLinearSystem(const QVector<QVector<double>>& A, const QVector<double>& b);
};

// ===========================================================================
// FittingDataLoadDialog: 数据加载配置弹窗
// ===========================================================================
class FittingDataLoadDialog : public QDialog {
    Q_OBJECT
public:
    FittingDataLoadDialog(const QList<QStringList>& previewData, QWidget* parent=nullptr);
    int getTimeColumnIndex() const;
    int getPressureColumnIndex() const;
    int getDerivativeColumnIndex() const;
    int getSkipRows() const;
    int getPressureDataType() const;
private slots:
    void validateSelection();
private:
    QTableWidget* m_previewTable;
    QComboBox* m_comboTime;
    QComboBox* m_comboPressure;
    QComboBox* m_comboDeriv;
    QComboBox* m_comboSkipRows;
    QComboBox* m_comboPressureType;
};

#endif // FITTINGWIDGET_H
