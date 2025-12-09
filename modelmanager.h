#ifndef MODELMANAGER_H
#define MODELMANAGER_H

#include <QObject>
#include <QWidget>
#include <QStackedWidget>
#include <QComboBox>
#include <QMap>
#include <QVector>
#include <tuple>

// 定义模型曲线数据类型: <时间(t), 压力(pD), 导数(dpD)>
typedef std::tuple<QVector<double>, QVector<double>, QVector<double>> ModelCurveData;

class ModelWidget1;
class ModelWidget2;
class ModelWidget3;

class ModelManager : public QObject
{
    Q_OBJECT

public:
    enum ModelType {
        InfiniteConductive = 0,    // 复合页岩油储层试井解释模型
        FiniteConductive = 1,      // 试井解释模型2
        SegmentedMultiCluster = 2  // 试井解释模型3
    };
    Q_ENUM(ModelType)

    explicit ModelManager(QWidget* parent = nullptr);
    ~ModelManager();

    void initializeModels(QWidget* parentWidget);
    QWidget* getMainWidget() const { return m_mainWidget; }
    void switchToModel(ModelType modelType);
    ModelType getCurrentModelType() const { return m_currentModelType; }

    static QString getModelTypeName(ModelType type);
    static QStringList getAvailableModelTypes();

    void setHighPrecision(bool high);
    QMap<QString, double> getDefaultParameters(ModelType type);

    // 统一计算接口，内部代理给具体的 ModelWidget
    ModelCurveData calculateTheoreticalCurve(ModelType type,
                                             const QMap<QString, double>& params,
                                             const QVector<double>& providedTime = QVector<double>());

    // 生成对数时间步长 (辅助工具)
    static QVector<double> generateLogTimeSteps(int count, double startExp, double endExp);

    // === 新增：实测数据持久化接口 ===
    void setObservedData(const QVector<double>& t, const QVector<double>& p, const QVector<double>& d);
    void getObservedData(QVector<double>& t, QVector<double>& p, QVector<double>& d) const;
    bool hasObservedData() const;

signals:
    void modelSwitched(ModelType newType, ModelType oldType);
    void calculationCompleted(const QString& title, const QMap<QString, double>& results);

private slots:
    void onModelTypeSelectionChanged(int index);
    // 接收子 Widget 计算完成信号并转发
    void onWidgetCalculationCompleted(const QString& t, const QMap<QString, double>& r);

private:
    void createMainWidget();
    void setupModelSelection();
    void connectModelSignals();

    QWidget* m_mainWidget;
    QComboBox* m_modelTypeCombo;
    QStackedWidget* m_modelStack;

    ModelWidget1* m_modelWidget1;
    ModelWidget2* m_modelWidget2;
    ModelWidget3* m_modelWidget3;

    ModelType m_currentModelType;

    // === 新增：缓存的实测数据 ===
    QVector<double> m_cachedObsTime;
    QVector<double> m_cachedObsPressure;
    QVector<double> m_cachedObsDerivative;
};

#endif // MODELMANAGER_H
