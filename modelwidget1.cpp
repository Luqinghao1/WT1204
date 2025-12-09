#include "modelwidget1.h"
#include "ui_modelwidget1.h"
#include "modelmanager.h"
#include "pressurederivativecalculator.h"

#include <Eigen/Dense>
#include <boost/math/special_functions/bessel.hpp>

#include <cmath>
#include <algorithm>
#include <QDebug>
#include <QMessageBox>
#include <QFileDialog>
#include <QTextStream>
#include <QDateTime>
#include <QCoreApplication>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

ModelWidget1::ModelWidget1(QWidget *parent) : QWidget(parent), ui(new Ui::ModelWidget1), m_highPrecision(true) {
    ui->setupUi(this);

    // 1. 初始化增强型图表控件
    initChart();

    // 2. 初始化敏感性分析颜色表 (红, 蓝, 绿, 紫, 橙, 青)
    m_colorList = { Qt::red, Qt::blue, QColor(0,180,0), Qt::magenta, QColor(255,140,0), Qt::cyan };

    // 3. 连接信号槽
    setupConnections();

    // 4. 初始化默认参数
    onResetParameters();
}

ModelWidget1::~ModelWidget1() { delete ui; }

/**
 * @brief 初始化图表
 * 修改：使其坐标系样式与 FittingWidget 完全一致（封闭边框、网格线样式）
 */
void ModelWidget1::initChart() {
    // 替换 UI 中的占位 Widget
    QVBoxLayout* layout = new QVBoxLayout(ui->chartContainer);
    layout->setContentsMargins(0,0,0,0);
    m_plot = new MouseZoom(this); // 使用 MouseZoom 类
    layout->addWidget(m_plot);

    m_plot->setBackground(Qt::white);
    m_plot->axisRect()->setBackground(Qt::white);

    // 配置对数坐标系 ticker
    QSharedPointer<QCPAxisTickerLog> logTicker(new QCPAxisTickerLog);
    m_plot->xAxis->setScaleType(QCPAxis::stLogarithmic); m_plot->xAxis->setTicker(logTicker);
    m_plot->yAxis->setScaleType(QCPAxis::stLogarithmic); m_plot->yAxis->setTicker(logTicker);

    // 科学计数法显示
    m_plot->xAxis->setNumberFormat("eb"); m_plot->xAxis->setNumberPrecision(0);
    m_plot->yAxis->setNumberFormat("eb"); m_plot->yAxis->setNumberPrecision(0);

    // 标签与网格
    QFont labelFont("Arial", 12, QFont::Bold);
    QFont tickFont("Arial", 12);
    m_plot->xAxis->setLabel("时间 Time (h)");
    m_plot->yAxis->setLabel("压力 & 导数 Pressure & Derivative (MPa)");
    m_plot->xAxis->setLabelFont(labelFont); m_plot->yAxis->setLabelFont(labelFont);
    m_plot->xAxis->setTickLabelFont(tickFont); m_plot->yAxis->setTickLabelFont(tickFont);

    // =========================================================
    // 封闭坐标系 (上下左右都有轴线)，与 FittingWidget 一致
    // =========================================================
    m_plot->xAxis2->setVisible(true); m_plot->yAxis2->setVisible(true);
    m_plot->xAxis2->setTickLabels(false); m_plot->yAxis2->setTickLabels(false);

    // 轴范围联动
    connect(m_plot->xAxis, SIGNAL(rangeChanged(QCPRange)), m_plot->xAxis2, SLOT(setRange(QCPRange)));
    connect(m_plot->yAxis, SIGNAL(rangeChanged(QCPRange)), m_plot->yAxis2, SLOT(setRange(QCPRange)));

    // 确保上右轴也是对数刻度
    m_plot->xAxis2->setScaleType(QCPAxis::stLogarithmic); m_plot->yAxis2->setScaleType(QCPAxis::stLogarithmic);
    m_plot->xAxis2->setTicker(logTicker); m_plot->yAxis2->setTicker(logTicker);

    // 网格线样式
    m_plot->xAxis->grid()->setVisible(true); m_plot->yAxis->grid()->setVisible(true);
    m_plot->xAxis->grid()->setSubGridVisible(true); m_plot->yAxis->grid()->setSubGridVisible(true);
    m_plot->xAxis->grid()->setPen(QPen(QColor(220, 220, 220), 1, Qt::SolidLine));
    m_plot->yAxis->grid()->setPen(QPen(QColor(220, 220, 220), 1, Qt::SolidLine));
    m_plot->xAxis->grid()->setSubGridPen(QPen(QColor(240, 240, 240), 1, Qt::DotLine));
    m_plot->yAxis->grid()->setSubGridPen(QPen(QColor(240, 240, 240), 1, Qt::DotLine));

    m_plot->xAxis->setRange(1e-3, 1e3); m_plot->yAxis->setRange(1e-3, 1e2);

    // 标题
    m_plot->plotLayout()->insertRow(0);
    m_plotTitle = new QCPTextElement(m_plot, "复合页岩油储层试井曲线", QFont("SimHei", 14, QFont::Bold));
    m_plot->plotLayout()->addElement(0, 0, m_plotTitle);

    // 开启图例
    m_plot->legend->setVisible(true);
    QFont legendFont = font();
    legendFont.setPointSize(9);
    m_plot->legend->setFont(legendFont);
    m_plot->legend->setBrush(QBrush(QColor(255, 255, 255, 200)));
}

void ModelWidget1::setupConnections() {
    connect(ui->calculateButton, &QPushButton::clicked, this, &ModelWidget1::onCalculateClicked);
    connect(ui->resetButton, &QPushButton::clicked, this, &ModelWidget1::onResetParameters);
    connect(ui->exportButton, &QPushButton::clicked, this, &ModelWidget1::onExportResults);
    connect(ui->resetViewButton, &QPushButton::clicked, this, &ModelWidget1::onResetView);
    connect(ui->fitToDataButton, &QPushButton::clicked, this, &ModelWidget1::onFitToData);
    connect(ui->chartSettingsButton, &QPushButton::clicked, this, &ModelWidget1::onChartSettings);

    // 联动逻辑：当 L 或 Lf 变化时，自动更新 LfD
    connect(ui->LEdit, &QLineEdit::editingFinished, this, &ModelWidget1::onDependentParamsChanged);
    connect(ui->LfEdit, &QLineEdit::editingFinished, this, &ModelWidget1::onDependentParamsChanged);
}

void ModelWidget1::setHighPrecision(bool high) { m_highPrecision = high; }

/**
 * @brief 解析输入文本
 * 支持 "10" 或 "10, 20, 30" (中文或英文逗号)
 */
QVector<double> ModelWidget1::parseInput(const QString& text) {
    QVector<double> values;
    QString cleanText = text;
    cleanText.replace("，", ","); // 兼容中文逗号
    QStringList parts = cleanText.split(",", Qt::SkipEmptyParts);
    for(const QString& part : parts) {
        bool ok;
        double v = part.trimmed().toDouble(&ok);
        if(ok) values.append(v);
    }
    // 如果解析失败，返回默认 0.0 防止崩溃
    if(values.isEmpty()) values.append(0.0);
    return values;
}

/**
 * @brief 设置输入框文本 (去除多余末尾0)
 */
void ModelWidget1::setInputText(QLineEdit* edit, double value) {
    if(!edit) return;
    edit->setText(QString::number(value, 'g', 8));
}

void ModelWidget1::onResetParameters() {
    // 基础参数
    setInputText(ui->phiEdit, 0.05);
    setInputText(ui->hEdit, 20.0);
    setInputText(ui->muEdit, 0.5);
    setInputText(ui->BEdit, 1.05);
    setInputText(ui->CtEdit, 5e-4);
    setInputText(ui->qEdit, 5.0);
    setInputText(ui->tEdit, 1000.0); // 新增：测试时间默认 1000h

    // 复合模型参数
    setInputText(ui->kfEdit, 1e-3);
    setInputText(ui->kmEdit, 1e-4);
    setInputText(ui->LEdit, 1000.0);
    setInputText(ui->LfEdit, 100.0);

    setInputText(ui->nfEdit, 4);
    setInputText(ui->rmDEdit, 4.0);
    setInputText(ui->omga1Edit, 0.4);
    setInputText(ui->omga2Edit, 0.08);
    setInputText(ui->remda1Edit, 0.001);

    setInputText(ui->cDEdit, 0);
    setInputText(ui->sEdit, 0);

    onDependentParamsChanged();
}

void ModelWidget1::onDependentParamsChanged() {
    // 简单的单值联动，如果输入了多个值，取第一个做参考
    double L = parseInput(ui->LEdit->text()).first();
    double Lf = parseInput(ui->LfEdit->text()).first();
    if (L > 1e-9) setInputText(ui->LfDEdit, Lf / L);
    else setInputText(ui->LfDEdit, 0.0);
}

void ModelWidget1::onResetView() {
    m_plot->rescaleAxes();
    m_plot->replot();
}

void ModelWidget1::onFitToData() {
    m_plot->rescaleAxes();
    // 确保对数轴下限为正
    if(m_plot->xAxis->range().lower <= 0) m_plot->xAxis->setRangeLower(1e-3);
    if(m_plot->yAxis->range().lower <= 0) m_plot->yAxis->setRangeLower(1e-3);
    m_plot->replot();
}

void ModelWidget1::onChartSettings() {
    // 调用通用的图表设置对话框
    ChartSetting1 dlg(m_plot, m_plotTitle, this);
    dlg.exec();
}

void ModelWidget1::onCalculateClicked() {
    ui->calculateButton->setEnabled(false);
    ui->calculateButton->setText("计算中...");
    QCoreApplication::processEvents();

    runCalculation();

    ui->calculateButton->setEnabled(true);
    ui->calculateButton->setText("开始计算");
    ui->exportButton->setEnabled(true);
    ui->resetViewButton->setEnabled(true);
    ui->fitToDataButton->setEnabled(true);
    ui->tabWidget->setCurrentIndex(0);
}

/**
 * @brief 核心计算流程 (包含敏感性分析逻辑)
 */
void ModelWidget1::runCalculation() {
    m_plot->clearGraphs(); // 清空旧曲线

    // 1. 收集所有参数的所有可能值
    QMap<QString, QVector<double>> rawParams;
    rawParams["phi"] = parseInput(ui->phiEdit->text());
    rawParams["h"] = parseInput(ui->hEdit->text());
    rawParams["mu"] = parseInput(ui->muEdit->text());
    rawParams["B"] = parseInput(ui->BEdit->text());
    rawParams["Ct"] = parseInput(ui->CtEdit->text());
    rawParams["q"] = parseInput(ui->qEdit->text());
    rawParams["t"] = parseInput(ui->tEdit->text()); // 读取测试时间

    rawParams["kf"] = parseInput(ui->kfEdit->text());
    rawParams["km"] = parseInput(ui->kmEdit->text());
    rawParams["L"] = parseInput(ui->LEdit->text());
    rawParams["Lf"] = parseInput(ui->LfEdit->text());
    rawParams["nf"] = parseInput(ui->nfEdit->text());
    rawParams["rmD"] = parseInput(ui->rmDEdit->text());
    rawParams["omega1"] = parseInput(ui->omga1Edit->text());
    rawParams["omega2"] = parseInput(ui->omga2Edit->text());
    rawParams["lambda1"] = parseInput(ui->remda1Edit->text());
    rawParams["cD"] = parseInput(ui->cDEdit->text());
    rawParams["S"] = parseInput(ui->sEdit->text());

    // 2. 识别是否进行敏感性分析
    // 规则：只要有一个参数输入的数值个数 > 1，则认为对该参数进行敏感性分析。
    QString sensitivityKey = "";
    QVector<double> sensitivityValues;

    for(auto it = rawParams.begin(); it != rawParams.end(); ++it) {
        // 时间参数 t 通常不用于敏感性分析（只影响x轴范围），跳过
        if(it.key() == "t") continue;

        if(it.value().size() > 1) {
            sensitivityKey = it.key();
            sensitivityValues = it.value();
            break;
        }
    }

    // 3. 准备基础参数字典 (所有参数取第一个值)
    QMap<QString, double> baseParams;
    for(auto it = rawParams.begin(); it != rawParams.end(); ++it) {
        baseParams[it.key()] = it.value().isEmpty() ? 0.0 : it.value().first();
    }
    // 补充额外参数
    baseParams["N"] = m_highPrecision ? 8.0 : 4.0;
    if(baseParams["L"] > 1e-9) baseParams["LfD"] = baseParams["Lf"] / baseParams["L"];
    else baseParams["LfD"] = 0;

    // 4. 生成时间步长
    // 根据输入的测试时间 t，确定生成曲线的 X 轴上限
    double maxTime = baseParams.value("t", 1000.0);
    if(maxTime < 1e-3) maxTime = 1000.0;
    // 生成从 10^-3 到 10^log(maxTime) 的对数时间步长
    QVector<double> t = ModelManager::generateLogTimeSteps(100, -3.0, log10(maxTime));

    // 5. 循环计算与绘图
    bool isSensitivity = !sensitivityKey.isEmpty();
    int iterations = isSensitivity ? sensitivityValues.size() : 1;

    // 限制最大颜色数，防止越界
    iterations = qMin(iterations, m_colorList.size());

    QString resultTextHeader = "计算完成\n";
    if(isSensitivity) resultTextHeader += QString("敏感性分析参数: %1\n").arg(sensitivityKey);

    for(int i = 0; i < iterations; ++i) {
        // 更新当前轮次的参数
        QMap<QString, double> currentParams = baseParams;
        double val = 0;

        if (isSensitivity) {
            val = sensitivityValues[i];
            currentParams[sensitivityKey] = val;
            // 如果改变了 L 或 Lf，需更新 LfD
            if (sensitivityKey == "L" || sensitivityKey == "Lf") {
                if(currentParams["L"] > 1e-9) currentParams["LfD"] = currentParams["Lf"] / currentParams["L"];
            }
        }

        // 计算
        ModelCurveData res = calculateTheoreticalCurve(currentParams, t);

        // 缓存最后一次结果供导出
        res_tD = std::get<0>(res);
        res_pD = std::get<1>(res);
        res_dpD = std::get<2>(res);

        // 确定颜色和图例名
        QColor curveColor = isSensitivity ? m_colorList[i] : Qt::red; // 正常模式压力为红
        QString legendName;
        if (isSensitivity) legendName = QString("%1 = %2").arg(sensitivityKey).arg(val);
        else legendName = "理论曲线";

        // 绘制
        plotCurve(res, legendName, curveColor, isSensitivity);
    }

    // 6. 更新结果文本框 (仅显示最后一次计算的前几行)
    QString resultText = resultTextHeader;
    resultText += "t(Time)\t\tDp(MPa)\t\tdDp(MPa)\n";
    for(int i=0; i<20 && i<res_pD.size(); ++i) {
        resultText += QString("%1\t%2\t%3\n").arg(res_tD[i],0,'e',4).arg(res_pD[i],0,'e',4).arg(res_dpD[i],0,'e',4);
    }
    ui->resultTextEdit->setText(resultText);

    // 自动适应视图
    onFitToData();
    emit calculationCompleted("Composite_Shale_Oil", baseParams);
}

/**
 * @brief 绘制单组曲线
 */
void ModelWidget1::plotCurve(const ModelCurveData& data, const QString& name, QColor color, bool isSensitivity) {
    const QVector<double>& t = std::get<0>(data);
    const QVector<double>& p = std::get<1>(data);
    const QVector<double>& d = std::get<2>(data);

    // 压力曲线
    QCPGraph* graphP = m_plot->addGraph();
    graphP->setData(t, p);
    graphP->setPen(QPen(color, 2, Qt::SolidLine)); // 压力始终实线

    // 导数曲线
    QCPGraph* graphD = m_plot->addGraph();
    graphD->setData(t, d);

    if (isSensitivity) {
        // 敏感性分析模式：导数用同色虚线
        graphD->setPen(QPen(color, 2, Qt::DashLine));
        // 图例只显示一条代表即可 (压力)
        graphP->setName(name);
        graphD->removeFromLegend();
    } else {
        // 普通模式：压力红实线，导数蓝实线 (参考 fittingwidget 标准)
        graphP->setPen(QPen(Qt::red, 2));
        graphP->setName("压力");

        graphD->setPen(QPen(Qt::blue, 2));
        graphD->setName("压力导数");
    }
}

void ModelWidget1::onExportResults() {
    if (res_tD.isEmpty()) return;
    QString path = QFileDialog::getSaveFileName(this, "导出CSV", "", "CSV Files (*.csv)");
    if (path.isEmpty()) return;
    QFile f(path);
    if (f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&f);
        out << "t,Dp,dDp\n";
        for (int i = 0; i < res_tD.size(); ++i) {
            double dp = (i < res_dpD.size()) ? res_dpD[i] : 0.0;
            out << res_tD[i] << "," << res_pD[i] << "," << dp << "\n";
        }
        f.close();
        QMessageBox::information(this, "导出成功", "文件已保存");
    }
}

// ==================================================================================
//  数学模型核心算法 (保持不变，省略具体实现细节以节省篇幅，请复制原有的数学函数实现)
// ==================================================================================

ModelCurveData ModelWidget1::calculateTheoreticalCurve(const QMap<QString, double>& params, const QVector<double>& providedTime)
{
    QVector<double> tPoints = providedTime;
    // 注意：这里的判空是为防止 providedTime 为空时的兜底，
    // 在 runCalculation 中我们已经根据 t 参数生成了 tPoints，所以一般不走这里。
    if (tPoints.isEmpty()) {
        tPoints = ModelManager::generateLogTimeSteps(100, -3.0, 3.0);
    }

    double phi = params.value("phi", 0.05);
    double mu = params.value("mu", 0.5);
    double B = params.value("B", 1.05);
    double Ct = params.value("Ct", 5e-4);
    double q = params.value("q", 5.0);
    double h = params.value("h", 20.0);
    double kf = params.value("kf", 1e-3);
    double L = params.value("L", 1000.0);

    QVector<double> tD_vec;
    tD_vec.reserve(tPoints.size());
    for(double t : tPoints) {
        double val = 14.4 * kf * t / (phi * mu * Ct * pow(L, 2));
        tD_vec.append(val);
    }

    QVector<double> PD_vec, Deriv_vec;
    auto func = std::bind(&ModelWidget1::flaplace_composite, this, std::placeholders::_1, std::placeholders::_2);
    calculatePDandDeriv(tD_vec, params, func, PD_vec, Deriv_vec);

    double factor = 1.842e-3 * q * mu * B / (kf * h);
    QVector<double> finalP(tPoints.size()), finalDP(tPoints.size());

    for(int i=0; i<tPoints.size(); ++i) {
        finalP[i] = factor * PD_vec[i];
        finalDP[i] = factor * Deriv_vec[i];
    }

    return std::make_tuple(tPoints, finalP, finalDP);
}

void ModelWidget1::calculatePDandDeriv(const QVector<double>& tD, const QMap<QString, double>& params,
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
        if (t <= 1e-12) { outPD[k] = 0; continue; }
        double pd_val = 0.0;
        for (int m = 1; m <= N; ++m) {
            double z = m * ln2 / t;
            double pf = laplaceFunc(z, params);
            if (std::isnan(pf) || std::isinf(pf)) pf = 0.0;
            pd_val += stefestCoefficient(m, N) * pf;
        }
        outPD[k] = pd_val * ln2 / t;
    }
    if (numPoints > 2) outDeriv = PressureDerivativeCalculator::calculateBourdetDerivative(tD, outPD, 0.1);
    else outDeriv.fill(0.0);
}

// 以下数学辅助函数请保持原样实现
double ModelWidget1::flaplace_composite(double z, const QMap<QString, double>& p) {
    double kf = p.value("kf"); double km = p.value("km"); double LfD = p.value("LfD");
    double rmD = p.value("rmD"); double omga1 = p.value("omega1"); double omga2 = p.value("omega2");
    double remda1 = p.value("lambda1");
    double M12 = kf / km; int nf = 4;
    QVector<double> xwD; for(int i=0; i<nf; ++i) xwD.append(-0.9 + i * (1.8) / (nf > 1 ? nf - 1 : 1));
    double temp = omga2; double fs1 = omga1 + remda1 * temp / (remda1 + z * temp); double fs2 = M12 * temp;
    double pf = PWD_inf(z, fs1, fs2, M12, LfD, rmD, nf, xwD);
    double CD = p.value("cD", 0.0); double S = p.value("S", 0.0);
    if (CD > 1e-12 || std::abs(S) > 1e-12) pf = (z * pf + S) / (z + CD * z * z * (z * pf + S));
    return pf;
}

double ModelWidget1::PWD_inf(double z, double fs1, double fs2, double M12, double LfD, double rmD, int nf, const QVector<double>& xwD) {
    using namespace boost::math;
    QVector<double> ywD(nf, 0.0);
    double gama1 = sqrt(z * fs1); double gama2 = sqrt(z * fs2);
    double arg_g2 = gama2 * rmD; double arg_g1 = gama1 * rmD;
    double k0_g2 = cyl_bessel_k(0, arg_g2); double k1_g2 = cyl_bessel_k(1, arg_g2);
    double k0_g1 = cyl_bessel_k(0, arg_g1); double k1_g1 = cyl_bessel_k(1, arg_g1);
    double Acup = M12 * gama1 * k1_g1 * k0_g2 - gama2 * k0_g1 * k1_g2;
    double i0_g1_s = scaled_besseli(0, arg_g1); double i1_g1_s = scaled_besseli(1, arg_g1);
    double Acdown_scaled = M12 * gama1 * i1_g1_s * k0_g2 + gama2 * i0_g1_s * k1_g2;
    if (std::abs(Acdown_scaled) < 1e-100) Acdown_scaled = 1e-100;
    double Ac_prefactor = Acup / Acdown_scaled;

    int size = nf + 1; Eigen::MatrixXd A_mat(size, size); Eigen::VectorXd b_vec(size); b_vec.setZero(); b_vec(nf) = 1.0;
    for (int i = 0; i < nf; ++i) {
        for (int j = 0; j < nf; ++j) {
            auto integrand = [&](double a) -> double {
                double dist = std::sqrt(std::pow(xwD[i] - xwD[j] - a, 2) + std::pow(ywD[i] - ywD[j], 2));
                double arg_dist = gama1 * dist; if (arg_dist < 1e-10) arg_dist = 1e-10;
                double term2 = 0.0; double exponent = arg_dist - arg_g1;
                if (exponent > -700.0) term2 = Ac_prefactor * scaled_besseli(0, arg_dist) * std::exp(exponent);
                return cyl_bessel_k(0, arg_dist) + term2;
            };
            double val = adaptiveGauss(integrand, -LfD, LfD, 1e-5, 0, 10);
            A_mat(i, j) = z * val / (M12 * z * 2 * LfD);
        }
    }
    for (int i = 0; i < nf; ++i) { A_mat(i, nf) = -1.0; A_mat(nf, i) = z; } A_mat(nf, nf) = 0.0;
    return A_mat.fullPivLu().solve(b_vec)(nf);
}

double ModelWidget1::scaled_besseli(int v, double x) {
    if (x < 0) x = -x; if (x > 600.0) return 1.0 / std::sqrt(2.0 * M_PI * x);
    return boost::math::cyl_bessel_i(v, x) * std::exp(-x);
}
double ModelWidget1::gauss15(std::function<double(double)> f, double a, double b) {
    static const double X[] = { 0.0, 0.201194, 0.394151, 0.570972, 0.724418, 0.848207, 0.937299, 0.987993 };
    static const double W[] = { 0.202578, 0.198431, 0.186161, 0.166269, 0.139571, 0.107159, 0.070366, 0.030753 };
    double h = 0.5 * (b - a); double c = 0.5 * (a + b); double s = W[0] * f(c);
    for (int i = 1; i < 8; ++i) { double dx = h * X[i]; s += W[i] * (f(c - dx) + f(c + dx)); }
    return s * h;
}
double ModelWidget1::adaptiveGauss(std::function<double(double)> f, double a, double b, double eps, int depth, int maxDepth) {
    double c = (a + b) / 2.0; double v1 = gauss15(f, a, b); double v2 = gauss15(f, a, c) + gauss15(f, c, b);
    if (depth >= maxDepth || std::abs(v1 - v2) < 1e-10 * std::abs(v2) + eps) return v2;
    return adaptiveGauss(f, a, c, eps/2, depth+1, maxDepth) + adaptiveGauss(f, c, b, eps/2, depth+1, maxDepth);
}
double ModelWidget1::stefestCoefficient(int i, int N) {
    double s = 0.0; int k1 = (i + 1) / 2; int k2 = std::min(i, N / 2);
    for (int k = k1; k <= k2; ++k) {
        double num = pow(k, N / 2.0) * factorial(2 * k);
        double den = factorial(N / 2 - k) * factorial(k) * factorial(k - 1) * factorial(i - k) * factorial(2 * k - i);
        if(den!=0) s += num/den;
    }
    return ((i + N / 2) % 2 == 0 ? 1.0 : -1.0) * s;
}
double ModelWidget1::factorial(int n) { if(n<=1)return 1; double r=1; for(int i=2;i<=n;++i)r*=i; return r; }
