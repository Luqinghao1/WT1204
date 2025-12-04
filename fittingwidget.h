#ifndef FITTINGWIDGET_H
#define FITTINGWIDGET_H

#include <QWidget>
#include <QFutureWatcher>
#include "modelmanager.h"
#include "qcustomplot.h"

namespace Ui {
class FittingWidget;
}

// 拟合参数结构体
struct FitParameter {
    QString name;
    QString displayName;
    double value;
    double min;
    double max;
    bool isFit;
};

class FittingWidget : public QWidget
{
    Q_OBJECT

public:
    explicit FittingWidget(QWidget *parent = nullptr);
    ~FittingWidget();

    void setModelManager(ModelManager* m);
    void setObservedData(const QVector<double>& t, const QVector<double>& p, const QVector<double>& d);

signals:
    // 信号：通知UI更新迭代曲线
    void sigIterationUpdated(double error, const QMap<QString,double>& currentParams,
                             const QVector<double>& t, const QVector<double>& p, const QVector<double>& d);
    void sigProgress(int percent);
    void fittingCompleted(ModelManager::ModelType type, const QMap<QString,double>& finalParams);

private slots:
    void on_btnLoadData_clicked();
    void on_btnRunFit_clicked();
    void on_btnStop_clicked();

    // 参数重置（用于初始化表格）
    void on_btnResetParams_clicked();

    // 导入模型（读取表格并绘图）
    void on_btnImportModel_clicked();

    void on_comboModelSelect_currentIndexChanged(int index);

    // 内部槽函数
    void onIterationUpdate(double err, const QMap<QString,double>& p,
                           const QVector<double>& t, const QVector<double>& p_curve, const QVector<double>& d_curve);
    void onFitFinished();

private:
    Ui::FittingWidget *ui;
    ModelManager* m_modelManager;
    QCustomPlot* m_plot;
    bool m_isFitting;
    bool m_stopRequested;

    // 观测数据
    QVector<double> m_obsTime;
    QVector<double> m_obsPressure;
    QVector<double> m_obsDerivative;

    // 拟合参数
    QList<FitParameter> m_parameters;

    // 异步任务监视器
    QFutureWatcher<void> m_watcher;

    // 辅助函数
    void setupPlot();
    void initModelCombo();
    void loadParamsToTable();
    void updateParamsFromTable();

    // 核心绘图更新逻辑
    void updateModelCurve();

    QString getParamDisplayName(const QString& key);
    QStringList getParamOrder(ModelManager::ModelType type); // 获取参数显示顺序
    QStringList parseLine(const QString& line);
    void plotCurves(const QVector<double>& t, const QVector<double>& p, const QVector<double>& d, bool isModel);

    // 拟合核心逻辑
    void runOptimizationTask(ModelManager::ModelType modelType, QList<FitParameter> fitParams);
    void runLevenbergMarquardtOptimization(ModelManager::ModelType modelType, QList<FitParameter> params);

    // 数学辅助
    QVector<double> calculateResiduals(const QMap<QString,double>& params, ModelManager::ModelType modelType);
    double calculateSumSquaredError(const QVector<double>& residuals);

    QVector<QVector<double>> computeJacobian(const QMap<QString,double>& params,
                                             const QVector<double>& baseResiduals,
                                             const QVector<int>& fitIndices,
                                             ModelManager::ModelType modelType,
                                             const QList<FitParameter>& currentFitParams);

    // 线性方程组求解
    QVector<double> solveLinearSystem(const QVector<QVector<double>>& A, const QVector<double>& b);
    double calculateError(const QMap<QString,double>& trialParams, ModelManager::ModelType modelType);
};

// 数据加载对话框类声明
class QTableWidget;
class QComboBox;

class FittingDataLoadDialog : public QDialog {
    Q_OBJECT
public:
    FittingDataLoadDialog(const QList<QStringList>& previewData, QWidget* parent=nullptr);
    int getTimeColumnIndex() const;
    int getPressureColumnIndex() const;
    int getDerivativeColumnIndex() const;
    int getSkipRows() const;
    // 新增：获取压力数据类型（0:原始压力, 1:压差）
    int getPressureDataType() const;

private slots:
    void validateSelection();
private:
    QTableWidget* m_previewTable;
    QComboBox* m_comboTime;
    QComboBox* m_comboPressure;
    QComboBox* m_comboDeriv;
    QComboBox* m_comboSkipRows;
    // 新增：压力类型选择框
    QComboBox* m_comboPressureType;
};

#endif // FITTINGWIDGET_H
