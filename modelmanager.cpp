
#include "modelmanager.h"
#include "modelwidget1.h"
#include "modelwidget2.h"
#include "modelwidget3.h"
#include "PressureDerivativeCalculator.h"

// 引入 Eigen
#include <Eigen/Dense>

// 引入 Boost Math
#include <boost/math/special_functions/bessel.hpp>

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QGroupBox>
#include <QDebug>
#include <cmath>
#include <QtMath>
#include <algorithm>
#include <functional>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ==================================================================================
//  ModelManager 构造与初始化
// ==================================================================================

ModelManager::ModelManager(QWidget* parent)
    : QObject(parent), m_mainWidget(nullptr), m_modelTypeCombo(nullptr), m_modelStack(nullptr)
    , m_modelWidget1(nullptr), m_modelWidget2(nullptr), m_modelWidget3(nullptr)
    , m_currentModelType(InfiniteConductive)
    , m_highPrecision(true)
{
}

ModelManager::~ModelManager() {}

void ModelManager::initializeModels(QWidget* parentWidget)
{
    if (!parentWidget) return;
    createMainWidget();
    setupModelSelection();

    m_modelStack = new QStackedWidget(m_mainWidget);
    m_modelWidget1 = new ModelWidget1(m_modelStack);
    m_modelWidget2 = new ModelWidget2(m_modelStack);
    m_modelWidget3 = new ModelWidget3(m_modelStack);

    m_modelStack->addWidget(m_modelWidget1);
    m_modelStack->addWidget(m_modelWidget2);
    m_modelStack->addWidget(m_modelWidget3);

    m_mainWidget->layout()->addWidget(m_modelStack);
    connectModelSignals();
    switchToModel(InfiniteConductive);

    if (parentWidget->layout()) parentWidget->layout()->addWidget(m_mainWidget);
    else {
        QVBoxLayout* layout = new QVBoxLayout(parentWidget);
        layout->addWidget(m_mainWidget);
        parentWidget->setLayout(layout);
    }
}

void ModelManager::createMainWidget()
{
    m_mainWidget = new QWidget();
    QVBoxLayout* mainLayout = new QVBoxLayout(m_mainWidget);
    mainLayout->setContentsMargins(10, 5, 10, 10);
    mainLayout->setSpacing(0);
    m_mainWidget->setLayout(mainLayout);
}

void ModelManager::setupModelSelection()
{
    if (!m_mainWidget) return;
    QGroupBox* selectionGroup = new QGroupBox("模型类型选择", m_mainWidget);
    selectionGroup->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    QHBoxLayout* selectionLayout = new QHBoxLayout(selectionGroup);
    selectionLayout->setContentsMargins(9, 9, 9, 9);
    selectionLayout->setSpacing(6);

    QLabel* typeLabel = new QLabel("模型类型:", selectionGroup);
    typeLabel->setMinimumWidth(100);
    m_modelTypeCombo = new QComboBox(selectionGroup);
    m_modelTypeCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    m_modelTypeCombo->setMinimumWidth(200);

    m_modelTypeCombo->addItem("复合页岩油储层试井解释模型");
    m_modelTypeCombo->addItem(getModelTypeName(FiniteConductive));
    m_modelTypeCombo->addItem(getModelTypeName(SegmentedMultiCluster));

    m_modelTypeCombo->setStyleSheet("color: black;");
    typeLabel->setStyleSheet("color: black;");
    selectionGroup->setStyleSheet("QGroupBox { color: black; font-weight: bold; }");

    connect(m_modelTypeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &ModelManager::onModelTypeSelectionChanged);
    selectionLayout->addWidget(typeLabel);
    selectionLayout->addWidget(m_modelTypeCombo);

    QVBoxLayout* mainLayout = qobject_cast<QVBoxLayout*>(m_mainWidget->layout());
    if (mainLayout) {
        mainLayout->addWidget(selectionGroup);
        mainLayout->setStretchFactor(selectionGroup, 0);
    }
}

void ModelManager::connectModelSignals()
{
    if (m_modelWidget1) connect(m_modelWidget1, &ModelWidget1::calculationCompleted, this, &ModelManager::onModel1CalculationCompleted);
    if (m_modelWidget2) connect(m_modelWidget2, &ModelWidget2::calculationCompleted, this, &ModelManager::onModel2CalculationCompleted);
    if (m_modelWidget3) connect(m_modelWidget3, &ModelWidget3::calculationCompleted, this, &ModelManager::onModel3CalculationCompleted);
}

void ModelManager::switchToModel(ModelType modelType)
{
    if (!m_modelStack) return;
    ModelType old = m_currentModelType;
    m_currentModelType = modelType;
    m_modelStack->setCurrentIndex((int)modelType);
    if (m_modelTypeCombo) m_modelTypeCombo->setCurrentIndex((int)modelType);
    emit modelSwitched(modelType, old);
}

void ModelManager::onModelTypeSelectionChanged(int index) { switchToModel((ModelType)index); }

QString ModelManager::getModelTypeName(ModelType type)
{
    switch (type) {
    case InfiniteConductive: return "复合页岩油储层试井解释模型";
    case FiniteConductive: return "有限导流双重孔隙介质页岩油藏渗流模型";
    case SegmentedMultiCluster: return "分段多簇压裂水平井双重孔隙介质页岩油藏渗流模型";
    default: return "未知模型";
    }
}

QStringList ModelManager::getAvailableModelTypes()
{
    return { getModelTypeName(InfiniteConductive), getModelTypeName(FiniteConductive), getModelTypeName(SegmentedMultiCluster) };
}

void ModelManager::onModel1CalculationCompleted(const QString &t, const QMap<QString, double> &r) { emit calculationCompleted(t, r); }
void ModelManager::onModel2CalculationCompleted(const QString &t, const QMap<QString, double> &r) { emit calculationCompleted(t, r); }
void ModelManager::onModel3CalculationCompleted(const QString &t, const QMap<QString, double> &r) { emit calculationCompleted(t, r); }

void ModelManager::setHighPrecision(bool high) {
    m_highPrecision = high;
}

// ==================================================================================
//  参数与计算路由
// ==================================================================================

QMap<QString, double> ModelManager::getDefaultParameters(ModelType type)
{
    QMap<QString, double> p;
    // 默认通用参数 (注意：InfiniteConductive 下面会覆盖 cD 和 S 为 0)
    p.insert("cD", 0.001);
    p.insert("S", 0.01);
    p.insert("N", 4.0);

    if (type == InfiniteConductive) { // 复合页岩油储层试井解释模型
        p.insert("kf", 1e-3);
        p.insert("km", 1e-4);
        p.insert("L", 1000.0);
        p.insert("Lf", 100.0);
        p.insert("LfD", 0.1);
        p.insert("rmD", 4.0);
        p.insert("omega1", 0.4);
        p.insert("omega2", 0.08);
        p.insert("lambda1", 1e-3);

        // 修改：与 ModelWidget1 界面默认值保持一致 (0, 0)
        p.insert("cD", 0.0);
        p.insert("S", 0.0);
    }
    else if (type == FiniteConductive) {
        p.insert("omega", 0.0155);
        p.insert("lambda", 0.083);
        p.insert("mf", 3.0);
        p.insert("nf", 5.0);
        p.insert("Xf", 193.0);
        p.insert("yy", 295.0);
        p.insert("y", 2758.0);
        p.insert("CFD", 0.9);
        p.insert("kpd", 0.04);
        p.insert("S", 0.81);
        p.insert("cD", 8.08e-8);
    }
    else if (type == SegmentedMultiCluster) {
        p.insert("omega1", 0.05);
        p.insert("omega2", 0.05);
        p.insert("lambda1", 1e-1);
        p.insert("lambda2", 1e-1);
        p.insert("mf1", 2.0);
        p.insert("mf2", 2.0);
        p.insert("nf", 5.0);
        p.insert("Xf1", 40.0);
        p.insert("Xf2", 40.0);
        p.insert("yy1", 70.0);
        p.insert("yy2", 70.0);
        p.insert("y", 800.0);
        p.insert("CFD1", 0.4);
        p.insert("CFD2", 0.4);
        p.insert("kpd", 0.045);
    }
    return p;
}

QVector<double> ModelManager::generateLogTimeSteps(int count, double startExp, double endExp) {
    QVector<double> t;
    t.reserve(count);
    for (int i = 0; i < count; ++i) {
        double exponent = startExp + (endExp - startExp) * i / (count - 1);
        t.append(pow(10.0, exponent));
    }
    return t;
}

ModelCurveData ModelManager::calculateTheoreticalCurve(ModelType type, const QMap<QString, double>& params, const QVector<double>& providedTime)
{
    QVector<double> tPoints = providedTime;
    if (tPoints.isEmpty()) {
        tPoints = generateLogTimeSteps(100, -3.0, 3.0);
    }

    switch(type) {
    case InfiniteConductive:
        return calculateCompositeModel(params, tPoints);
    case FiniteConductive:
        return calculateModel2(params, tPoints);
    case SegmentedMultiCluster:
        return calculateModel3(params, tPoints);
    default:
        return calculateCompositeModel(params, tPoints);
    }
}

// ==================================================================================
//  核心算法：Stehfest 数值反演通用框架
// ==================================================================================

void ModelManager::calculatePDandDeriv(const QVector<double>& tD, const QMap<QString, double>& params,
                                       std::function<double(double, const QMap<QString, double>&)> laplaceFunc,
                                       QVector<double>& outPD, QVector<double>& outDeriv)
{
    int numPoints = tD.size();
    outPD.resize(numPoints);
    outDeriv.resize(numPoints);

    int N_param = (int)params.value("N", 4);
    int N = m_highPrecision ? N_param : 4;
    if (N % 2 != 0) N = 4;

    double ln2 = log(2.0);

    for (int k = 0; k < numPoints; ++k) {
        double t = tD[k];
        if (t <= 1e-12) {
            outPD[k] = 0;
            continue;
        }

        double pd_val = 0.0;
        for (int m = 1; m <= N; ++m) {
            double z = m * ln2 / t;
            double pf = laplaceFunc(z, params);
            if (std::isnan(pf) || std::isinf(pf)) pf = 0.0; // 保护
            double Vi = stefestCoefficient(m, N);
            pd_val += Vi * pf;
        }
        outPD[k] = pd_val * ln2 / t;
    }

    if (numPoints > 2) {
        outDeriv = PressureDerivativeCalculator::calculateBourdetDerivative(tD, outPD, 0.1);
    } else {
        outDeriv.fill(0.0);
    }
}

// ==================================================================================
//  模型1：复合页岩油储层试井解释模型 (Composite Shale Oil) - 修复溢出
// ==================================================================================

ModelCurveData ModelManager::calculateCompositeModel(const QMap<QString, double>& params, const QVector<double>& tPoints)
{
    // 参数
    double phi = 0.05;
    double mu = 0.5;
    double B = 1.05;
    double Ct = 5e-4;
    double q = 5.0;
    double h = 20.0;

    double kf = params.value("kf", 1e-3);
    double L = params.value("L", 1000.0);

    QVector<double> tD_vec;
    tD_vec.reserve(tPoints.size());
    for(double t : tPoints) {
        double val = 14.4 * kf * t / (phi * mu * Ct * pow(L, 2));
        tD_vec.append(val);
    }

    QVector<double> PD_vec, Deriv_vec;
    auto func = std::bind(&ModelManager::flaplace_composite, this, std::placeholders::_1, std::placeholders::_2);
    calculatePDandDeriv(tD_vec, params, func, PD_vec, Deriv_vec);

    double factor = 1.842e-3 * q * mu * B / (kf * h);

    QVector<double> finalP, finalDP;
    finalP.resize(tPoints.size());
    finalDP.resize(tPoints.size());

    double gamaD = 0.02;

    for(int i=0; i<tPoints.size(); ++i) {
        double pd_raw = PD_vec[i];
        double dpd_raw = Deriv_vec[i];

        double pd_corr = pd_raw;
        double dpd_corr = dpd_raw;

        if (gamaD != 0) {
            pd_corr = -1.0/gamaD * log(1.0 - gamaD * pd_raw);
            if ((1.0 - gamaD * pd_raw) > 1e-9)
                dpd_corr = dpd_raw / (1.0 - gamaD * pd_raw);
        }

        finalP[i] = factor * pd_corr;
        finalDP[i] = factor * dpd_corr;
    }

    return std::make_tuple(tPoints, finalP, finalDP);
}

double ModelManager::flaplace_composite(double z, const QMap<QString, double>& p) {
    double kf = p.value("kf");
    double km = p.value("km");
    double LfD = p.value("LfD");
    double rmD = p.value("rmD");
    double omga1 = p.value("omega1");
    double omga2 = p.value("omega2");
    double remda1 = p.value("lambda1");

    double M12 = kf / km;
    int nf = 4;

    QVector<double> xwD;
    for(int i=0; i<nf; ++i) {
        double val = -0.9 + i * (1.8) / (nf > 1 ? nf - 1 : 1);
        xwD.append(val);
    }

    double temp = omga2;
    double fs1 = omga1 + remda1 * temp / (remda1 + z * temp);
    double fs2 = M12 * temp;

    double pf = PWD_inf(z, fs1, fs2, M12, LfD, rmD, nf, xwD);

    double CD = p.value("cD", 0.0); // 确保读取时也有默认值
    double S = p.value("S", 0.0);

    if (CD > 1e-12 || std::abs(S) > 1e-12) {
        pf = (z * pf + S) / (z + CD * z * z * (z * pf + S));
    }

    return pf;
}

// --------------------------------------------------------------------------------------
// 修复核心：PWD_inf 使用缩放 Bessel 函数避免溢出
// --------------------------------------------------------------------------------------
double ModelManager::PWD_inf(double z, double fs1, double fs2, double M12, double LfD, double rmD, int nf, const QVector<double>& xwD)
{
    QVector<double> ywD(nf, 0.0);

    double gama1 = sqrt(z * fs1);
    double gama2 = sqrt(z * fs2);

    // Boost 的 K0/K1 计算
    using namespace boost::math;
    double arg_g2 = gama2 * rmD;
    double arg_g1 = gama1 * rmD;

    double k0_g2_val = cyl_bessel_k(0, arg_g2);
    double k1_g2_val = cyl_bessel_k(1, arg_g2);
    double k0_g1_val = cyl_bessel_k(0, arg_g1);
    double k1_g1_val = cyl_bessel_k(1, arg_g1);

    double Acup = M12 * gama1 * k1_g1_val * k0_g2_val - gama2 * k0_g1_val * k1_g2_val;

    // 计算分母的缩放项 i0_scaled = I0(x) * exp(-x)
    double i0_g1_s = scaled_besseli(0, arg_g1);
    double i1_g1_s = scaled_besseli(1, arg_g1);

    double Acdown_scaled = M12 * gama1 * i1_g1_s * k0_g2_val + gama2 * i0_g1_s * k1_g2_val;

    if (std::abs(Acdown_scaled) < 1e-100) Acdown_scaled = 1e-100;

    double Ac_prefactor = Acup / Acdown_scaled; // 实际 Ac = Ac_prefactor * exp(-arg_g1)

    // 构建矩阵
    int size = nf + 1;
    Eigen::MatrixXd A_mat(size, size);
    Eigen::VectorXd b_vec(size);
    b_vec.setZero();
    b_vec(nf) = 1.0;

    for (int i = 0; i < nf; ++i) {
        for (int j = 0; j < nf; ++j) {
            auto integrand = [&](double a) -> double {
                double dist_sq = std::pow(xwD[i] - xwD[j] - a, 2) + std::pow(ywD[i] - ywD[j], 2);
                double dist = std::sqrt(dist_sq);
                double arg_dist = gama1 * dist;

                if (arg_dist < 1e-10) arg_dist = 1e-10;

                double term1 = cyl_bessel_k(0, arg_dist);

                // 计算第二项：Ac * I0
                double term2 = 0.0;
                double exponent = arg_dist - arg_g1; // g1*dist - g1*rm

                if (exponent > -700.0) {
                    double i0_s_dist = scaled_besseli(0, arg_dist);
                    term2 = Ac_prefactor * i0_s_dist * std::exp(exponent);
                }

                return term1 + term2;
            };

            double val = adaptiveGauss(integrand, -LfD, LfD, 1e-5, 0, 10);
            double pfD = val / (M12 * z * 2 * LfD);
            A_mat(i, j) = z * pfD;
        }
    }

    for (int i = 0; i < nf; ++i) {
        A_mat(i, nf) = -1.0;
        A_mat(nf, i) = z;
    }
    A_mat(nf, nf) = 0.0;

    Eigen::VectorXd x_res = A_mat.fullPivLu().solve(b_vec);
    return x_res(nf);
}

// --------------------------------------------------------------------------------------
// Bessel 函数辅助：带缩放的 Bessel I，防止溢出
// --------------------------------------------------------------------------------------
double ModelManager::scaled_besseli(int v, double x) {
    if (x < 0) x = -x;
    if (x > 600.0) {
        return 1.0 / std::sqrt(2.0 * M_PI * x);
    }
    return boost::math::cyl_bessel_i(v, x) * std::exp(-x);
}

// --------------------------------------------------------------------------------------
// Model 2 & 3 及其他
// --------------------------------------------------------------------------------------

ModelCurveData ModelManager::calculateModel2(const QMap<QString, double>& params, const QVector<double>& tPoints)
{
    QVector<double> pd, dpd;
    auto func = std::bind(&ModelManager::flaplace2, this, std::placeholders::_1, std::placeholders::_2);
    calculatePDandDeriv(tPoints, params, func, pd, dpd);

    double kpd = params.value("kpd", 0.0);
    if (std::abs(kpd) > 1e-5) {
        for(int i=0; i<pd.size(); ++i) {
            double raw_p = pd[i];
            pd[i] = -1.0 / kpd * log(1.0 - kpd * raw_p);
            if ((1.0 - kpd * raw_p) > 1e-9)
                dpd[i] = dpd[i] / (1.0 - kpd * raw_p);
        }
    }
    return std::make_tuple(tPoints, pd, dpd);
}

ModelCurveData ModelManager::calculateModel3(const QMap<QString, double>& params, const QVector<double>& tPoints)
{
    return calculateCompositeModel(params, tPoints);
}

double ModelManager::flaplace2(double z, const QMap<QString, double>& p) {
    int mf = (int)p.value("mf", 3);
    int nf = (int)p.value("nf", 5);
    double omega = p.value("omega", 0.0155);
    double lambda = p.value("lambda", 0.083);
    double Xf = p.value("Xf", 193.0);
    double yy = p.value("yy", 295.0);
    double y = p.value("y", 2758.0);
    double CFD = p.value("CFD", 0.9);

    if(mf<=0 || nf<=0) return 0.0;
    int sz = mf * nf;
    double deltaL = Xf / (nf * y);

    QVector<QVector<double>> I(sz+mf+1, QVector<double>(sz+mf+1));
    QVector<double> F(sz+mf+1, 0);

    for(int i=1; i<=mf; ++i) {
        for(int j=1; j<=nf; ++j) {
            int r = (i-1)*nf + j - 1;
            for(int k=1; k<=mf; ++k) {
                for(int v=1; v<=nf; ++v) {
                    double val = integralBesselK0(
                        ((2*v-2*nf-1)/(2.0*nf)*Xf)/y, (yy+(y-2*yy)/(mf-1)*(k-1))/y, (yy+(y-2*yy)/(mf-1)*(i-1))/y,
                        z*(omega*z*(1-omega)+lambda)/(lambda+(1-omega)*z), ((j-nf-1)/(double)nf*Xf)/y, ((j-nf)/(double)nf*Xf)/y
                        );
                    if(i==k) val -= (j==v ? M_PI/(CFD*8)*deltaL*deltaL : M_PI/CFD*std::abs(j-v)*deltaL*deltaL);
                    I[r][(k-1)*nf + v - 1] = val;
                }
            }
            I[r][sz] = (2*M_PI/CFD) * (((j-nf)/(double)nf*Xf/y) - ((j-nf-1)/(double)nf*Xf/y));
            I[r][sz+mf] = -1.0;
        }
    }
    for(int i=1; i<=mf; ++i) {
        for(int j=(i-1)*nf; j<i*nf; ++j) I[sz+i-1][j] = deltaL;
        I[sz+i-1][sz+i-1] = -1.0;
    }
    for(int i=sz; i<sz+mf; ++i) I[sz+mf][i] = 1.0;

    F[sz+mf] = 1.0/z;
    QVector<double> res = solveLinearSystem(I, F);
    if(res.size() <= sz+mf) return 0.0;

    return res[sz+mf];
}

double ModelManager::flaplace1(double, const QMap<QString, double>&) {
    return 0.0; // Placeholder
}

// 辅助函数：积分 BesselK0
double ModelManager::integralBesselK0(double XDkv, double YDkv, double yDij, double fz, double a, double b) {
    auto func = [=](double xwD) {
        double dist_x = XDkv - xwD;
        double dist_y_sq = (YDkv - yDij) * (YDkv - yDij);
        double arg = sqrt(dist_x * dist_x + dist_y_sq) * sqrt(fz);
        return besselK0(arg);
    };
    return adaptiveGauss(func, a, b, 1e-6, 0, 10);
}

double ModelManager::besselK0(double x) {
    if (x <= 1e-15) return 50.0;
    return boost::math::cyl_bessel_k(0, x);
}

// ==================================================================================
//  数值计算辅助方法 (积分与线性方程组)
// ==================================================================================

static const double GL15_X[] = { 0.0, 0.2011940939974345, 0.3941513470775634, 0.5709721726085388, 0.7244177313601701, 0.8482065834104272, 0.9372985251687639, 0.9879925180204854 };
static const double GL15_W[] = { 0.2025782419255613, 0.1984314853271116, 0.1861610000155622, 0.1662692058169939, 0.1395706779049514, 0.1071592204671719, 0.0703660474881081, 0.0307532419961173 };

double ModelManager::gauss15(std::function<double(double)> f, double a, double b) {
    double halfLen = 0.5 * (b - a);
    double center = 0.5 * (a + b);
    double sum = GL15_W[0] * f(center);
    for (int i = 1; i < 8; ++i) {
        double dx = halfLen * GL15_X[i];
        sum += GL15_W[i] * (f(center - dx) + f(center + dx));
    }
    return sum * halfLen;
}

double ModelManager::adaptiveGauss(std::function<double(double)> f, double a, double b, double eps, int depth, int maxDepth) {
    double c = (a + b) / 2.0;
    double v1 = gauss15(f, a, b);
    double v2 = gauss15(f, a, c) + gauss15(f, c, b);
    if (depth >= maxDepth) return v2;
    if (std::abs(v1 - v2) < 1e-10 * std::abs(v2) + eps) return v2;
    return adaptiveGauss(f, a, c, eps/2, depth+1, maxDepth) + adaptiveGauss(f, c, b, eps/2, depth+1, maxDepth);
}

// 简单的 Gaussian Elimination 用于 Model 2 (旧代码兼容)
QVector<double> ModelManager::solveLinearSystem(QVector<QVector<double>> A, QVector<double> b) {
    int n = b.size();
    for(int i=0; i<n; ++i) A[i].append(b[i]);
    for (int k = 0; k < n - 1; ++k) {
        if (std::abs(A[k][k]) < 1e-12) continue;
        for (int i = k + 1; i < n; ++i) {
            double factor = A[i][k] / A[k][k];
            for (int j = k; j < n + 1; ++j) A[i][j] -= factor * A[k][j];
        }
    }
    QVector<double> x(n);
    if (std::abs(A[n-1][n-1]) > 1e-12) x[n-1] = A[n-1][n] / A[n-1][n-1];
    for (int i = n - 2; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < n; ++j) sum += A[i][j] * x[j];
        if (std::abs(A[i][i]) > 1e-12) x[i] = (A[i][n] - sum) / A[i][i];
    }
    return x;
}

double ModelManager::stefestCoefficient(int i, int N) {
    double sum = 0.0;
    int k_start = (i + 1) / 2;
    int k_end = std::min(i, N / 2);
    for (int k = k_start; k <= k_end; ++k) {
        double num = pow(k, N / 2.0) * factorial(2 * k);
        double den = factorial(N / 2 - k) * factorial(k) * factorial(k - 1) * factorial(i - k) * factorial(2 * k - i);
        if(den != 0) sum += num / den;
    }
    return ((i + N / 2) % 2 == 0 ? 1.0 : -1.0) * sum;
}

double ModelManager::factorial(int n) {
    if (n <= 1) return 1.0;
    double res = 1.0;
    for (int i = 2; i <= n; ++i) res *= i;
    return res;
}

double ModelManager::f_function(int j, int nf, double Xf, double y)
{
    double xij = (j - nf - 1) / (double)nf * Xf;
    double xDij = xij / y;
    double xij1 = (j - nf) / (double)nf * Xf;
    double xDij1 = xij1 / y;
    return xDij1 - xDij;
}

double ModelManager::e_function(double, int, int, int, int, int, int, double, double, double, double, double)
{
    return 0.0; // Placeholder for legacy logic if needed
}


