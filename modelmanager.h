#ifndef MODELMANAGER_H
#define MODELMANAGER_H

#include <QObject>
#include <QWidget>
#include <QStackedWidget>
#include <QComboBox>
#include <QMap>
#include <QVector>
#include <tuple>
#include <functional>

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
        FiniteConductive = 1,      // 有限导流
        SegmentedMultiCluster = 2  // 分段多簇
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

    ModelCurveData calculateTheoreticalCurve(ModelType type,
                                             const QMap<QString, double>& params,
                                             const QVector<double>& providedTime = QVector<double>());

signals:
    void modelSwitched(ModelType newType, ModelType oldType);
    void calculationCompleted(const QString& title, const QMap<QString, double>& results);

private slots:
    void onModelTypeSelectionChanged(int index);
    void onModel1CalculationCompleted(const QString& t, const QMap<QString, double>& r);
    void onModel2CalculationCompleted(const QString& t, const QMap<QString, double>& r);
    void onModel3CalculationCompleted(const QString& t, const QMap<QString, double>& r);

private:
    void createMainWidget();
    void setupModelSelection();
    void connectModelSignals();

    // 具体的模型计算分支
    ModelCurveData calculateCompositeModel(const QMap<QString, double>& params, const QVector<double>& tPoints);
    ModelCurveData calculateModel2(const QMap<QString, double>& params, const QVector<double>& tPoints);
    ModelCurveData calculateModel3(const QMap<QString, double>& params, const QVector<double>& tPoints);

    void calculatePDandDeriv(const QVector<double>& tD, const QMap<QString, double>& params,
                             std::function<double(double, const QMap<QString, double>&)> laplaceFunc,
                             QVector<double>& outPD, QVector<double>& outDeriv);

    // 数学函数 - 复合模型 (Model 1)
    double flaplace_composite(double z, const QMap<QString, double>& p);
    double PWD_inf(double z, double fs1, double fs2, double M12, double LfD, double rmD, int nf, const QVector<double>& xwD);

    // 数学函数 - 有限导流模型 (Model 2) & 辅助函数
    double flaplace1(double z, const QMap<QString, double>& p); // 保留旧接口声明以防万一
    double flaplace2(double z, const QMap<QString, double>& p);

    // Bessel 函数封装
    double besselK0(double x);
    double scaled_besseli(int v, double x); // 新增：缩放Bessel I 防止溢出

    // 线性方程组求解
    QVector<double> solveLinearSystem(QVector<QVector<double>> A, QVector<double> b);

    // 通用积分与算法
    double e_function(double z, int i, int j, int k, int v, int mf, int nf, double omega, double lambda, double Xf, double yy, double y);
    double f_function(int j, int nf, double Xf, double y);
    double stefestCoefficient(int i, int N);
    double factorial(int n);
    double adaptiveGauss(std::function<double(double)> f, double a, double b, double eps, int depth, int maxDepth);
    double gauss15(std::function<double(double)> f, double a, double b);
    double integralBesselK0(double XDkv, double YDkv, double yDij, double fz, double a, double b);
    QVector<double> generateLogTimeSteps(int count, double startExp, double endExp);

    QWidget* m_mainWidget;
    QComboBox* m_modelTypeCombo;
    QStackedWidget* m_modelStack;
    ModelWidget1* m_modelWidget1;
    ModelWidget2* m_modelWidget2;
    ModelWidget3* m_modelWidget3;
    ModelType m_currentModelType;
    bool m_highPrecision;
};

#endif // MODELMANAGER_H
