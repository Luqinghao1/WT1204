#include "modelwidget1.h"
#include "ui_modelwidget1.h"
#include "modelmanager.h"
#include <cmath>
#include <algorithm>
#include <QDebug>
#include <QMessageBox>
#include <QFileDialog>
#include <QTextStream>
#include <QDateTime>
#include <QCoreApplication>

// =========================================================
// LogLogChartWidget 实现 (绘图组件)
// =========================================================

LogLogChartWidget::LogLogChartWidget(QWidget *parent)
    : QWidget(parent)
    , m_xMin(1e-3), m_xMax(1e3)
    , m_yMin(1e-3), m_yMax(1e2)
    , m_hasData(false), m_showOriginalData(true), m_isDragging(false)
{
    setMinimumSize(600, 400);
    setStyleSheet("QWidget { background-color: white; border: 1px solid gray; }");
    setMouseTracking(true);
}

void LogLogChartWidget::setData(const QVector<double>& xData, const QVector<double>& yData1,
                                const QVector<double>& yData2, const QVector<double>& xData2)
{
    m_xData = xData; m_yData1 = yData1; m_yData2 = yData2;
    m_xData2 = xData2.isEmpty() ? xData : xData2;
    m_hasData = !xData.isEmpty();
    if (m_hasData) autoFitData();
    update();
}

void LogLogChartWidget::clearData() {
    m_hasData = false; m_xData.clear(); m_yData1.clear(); m_yData2.clear(); m_xData2.clear();
    resetView(); update();
}

void LogLogChartWidget::resetView() {
    if (m_hasData) autoFitData();
    else { m_xMin = 1e-3; m_xMax = 1e3; m_yMin = 1e-3; m_yMax = 1e2; }
    update();
}

void LogLogChartWidget::autoFitData() {
    if (!m_hasData) return;
    QVector<double> allX, allY;
    for (double x : m_xData) if (x > 0 && std::isfinite(x)) allX.append(x);
    for (double y : m_yData1) if (y > 0 && std::isfinite(y)) allY.append(y);
    for (double y : m_yData2) if (y > 0 && std::isfinite(y)) allY.append(y);
    if (allX.isEmpty() || allY.isEmpty()) return;

    auto [xMinIt, xMaxIt] = std::minmax_element(allX.begin(), allX.end());
    auto [yMinIt, yMaxIt] = std::minmax_element(allY.begin(), allY.end());

    double logXMin = log10(*xMinIt), logXMax = log10(*xMaxIt);
    double logYMin = log10(*yMinIt), logYMax = log10(*yMaxIt);
    double xMargin = (logXMax - logXMin) * 0.05;
    double yMargin = (logYMax - logYMin) * 0.05;

    m_xMin = pow(10.0, logXMin - xMargin); m_xMax = pow(10.0, logXMax + xMargin);
    m_yMin = pow(10.0, logYMin - yMargin); m_yMax = pow(10.0, logYMax + yMargin);
}

void LogLogChartWidget::mousePressEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        m_isDragging = true; m_lastMousePos = event->pos();
        setCursor(Qt::ClosedHandCursor);
    }
}

void LogLogChartWidget::mouseMoveEvent(QMouseEvent *event) {
    if (m_isDragging) {
        QPoint delta = event->pos() - m_lastMousePos; m_lastMousePos = event->pos();
        QRect plotRect = rect().adjusted(80, 50, -50, -80);
        double logXRange = log10(m_xMax) - log10(m_xMin);
        double logYRange = log10(m_yMax) - log10(m_yMin);
        m_xMin *= pow(10.0, -delta.x() * logXRange / plotRect.width());
        m_xMax *= pow(10.0, -delta.x() * logXRange / plotRect.width());
        m_yMin *= pow(10.0, delta.y() * logYRange / plotRect.height());
        m_yMax *= pow(10.0, delta.y() * logYRange / plotRect.height());
        update();
    }
}

void LogLogChartWidget::mouseReleaseEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) { m_isDragging = false; setCursor(Qt::OpenHandCursor); }
}

void LogLogChartWidget::wheelEvent(QWheelEvent *event) {
    double factor = (event->angleDelta().y() > 0) ? 0.9 : 1.1;
    double logXCenter = log10(m_xMin * m_xMax) / 2.0;
    double logXHalf = (log10(m_xMax) - log10(m_xMin)) / 2.0 * factor;
    m_xMin = pow(10.0, logXCenter - logXHalf); m_xMax = pow(10.0, logXCenter + logXHalf);

    double logYCenter = log10(m_yMin * m_yMax) / 2.0;
    double logYHalf = (log10(m_yMax) - log10(m_yMin)) / 2.0 * factor;
    m_yMin = pow(10.0, logYCenter - logYHalf); m_yMax = pow(10.0, logYCenter + logYHalf);
    update();
}

void LogLogChartWidget::paintEvent(QPaintEvent *) {
    QPainter painter(this); painter.setRenderHint(QPainter::Antialiasing);
    QRect plotRect = rect().adjusted(80, 50, -50, -80);
    painter.fillRect(rect(), Qt::white); painter.fillRect(plotRect, QColor(250, 250, 250));
    drawAxis(painter, plotRect);
    if (m_hasData) { drawData(painter, plotRect); drawLegend(painter, plotRect); }
}

void LogLogChartWidget::drawAxis(QPainter& painter, const QRect& plotRect) {
    painter.setPen(QPen(Qt::black, 2)); painter.drawRect(plotRect);
    painter.setPen(QPen(Qt::lightGray, 1)); painter.setFont(QFont("Arial", 9));

    for (int exp = floor(log10(m_xMin)); exp <= ceil(log10(m_xMax)); exp++) {
        double x = pow(10.0, exp);
        if (x >= m_xMin && x <= m_xMax) {
            QPointF p = dataToPixel(x, m_yMin, plotRect);
            painter.drawLine(p.x(), plotRect.bottom(), p.x(), plotRect.top());
            painter.setPen(Qt::black);
            painter.drawText(QRect(p.x()-25, plotRect.bottom()+5, 50, 20), Qt::AlignCenter, QString("1e%1").arg(exp));
            painter.setPen(Qt::lightGray);
        }
    }
    for (int exp = floor(log10(m_yMin)); exp <= ceil(log10(m_yMax)); exp++) {
        double y = pow(10.0, exp);
        if (y >= m_yMin && y <= m_yMax) {
            QPointF p = dataToPixel(m_xMin, y, plotRect);
            painter.drawLine(plotRect.left(), p.y(), plotRect.right(), p.y());
            painter.setPen(Qt::black);
            painter.drawText(QRect(plotRect.left()-50, p.y()-10, 45, 20), Qt::AlignRight|Qt::AlignVCenter, QString("1e%1").arg(exp));
            painter.setPen(Qt::lightGray);
        }
    }
    // 轴标签
    painter.setPen(Qt::black); painter.setFont(QFont("Arial", 11, QFont::Bold));
    painter.drawText(plotRect.center().x()-20, plotRect.bottom()+40, "t (h)");
    painter.save();
    painter.translate(plotRect.left()-60, plotRect.center().y());
    painter.rotate(-90);
    painter.drawText(-50, 0, "Dp & dDp (MPa)");
    painter.restore();
}

void LogLogChartWidget::drawData(QPainter& painter, const QRect& plotRect) {
    auto drawCurve = [&](const QVector<double>& x, const QVector<double>& y, QColor color) {
        if (x.isEmpty() || y.isEmpty()) return;
        painter.setPen(QPen(color, 2));
        QVector<QPointF> points;
        for (int i=0; i<qMin(x.size(), y.size()); ++i)
            if (x[i]>0 && y[i]>0) points.append(dataToPixel(x[i], y[i], plotRect));
        for (int i=1; i<points.size(); ++i)
            if (plotRect.contains(points[i-1].toPoint()) || plotRect.contains(points[i].toPoint()))
                painter.drawLine(points[i-1], points[i]);
        if (m_showOriginalData) {
            painter.setPen(QPen(color, 1)); painter.setBrush(color);
            for (auto& p : points) if (plotRect.contains(p.toPoint())) painter.drawEllipse(p, 2, 2);
        }
    };
    drawCurve(m_xData, m_yData1, Qt::red); drawCurve(m_xData2, m_yData2, Qt::blue);
}

void LogLogChartWidget::drawLegend(QPainter& painter, const QRect& plotRect) {
    painter.setFont(QFont("Arial", 10)); int x = plotRect.right()-100, y = plotRect.top()+20;
    painter.setPen(QPen(Qt::red, 2)); painter.drawLine(x, y, x+20, y); painter.setPen(Qt::black); painter.drawText(x+25, y+5, "压力");
    if (!m_yData2.isEmpty()) {
        y += 20; painter.setPen(QPen(Qt::blue, 2)); painter.drawLine(x, y, x+20, y); painter.setPen(Qt::black); painter.drawText(x+25, y+5, "压力导数");
    }
}

QPointF LogLogChartWidget::dataToPixel(double x, double y, const QRect& plotRect) {
    x = qMax(x, 1e-20); y = qMax(y, 1e-20);
    double lx = log10(x), ly = log10(y), lxmin = log10(qMax(m_xMin, 1e-20)), lxmax = log10(qMax(m_xMax, 1e-20)), lymin = log10(qMax(m_yMin, 1e-20)), lymax = log10(qMax(m_yMax, 1e-20));
    return QPointF(plotRect.left() + (lx - lxmin)/(lxmax - lxmin)*plotRect.width(), plotRect.bottom() - (ly - lymin)/(lymax - lymin)*plotRect.height());
}

// =========================================================
// ModelWidget1 主逻辑 (复合页岩油储层试井解释模型)
// =========================================================

ModelWidget1::ModelWidget1(QWidget *parent) : QWidget(parent), ui(new Ui::ModelWidget1) {
    ui->setupUi(this);
    m_chartWidget = new LogLogChartWidget(this);
    QVBoxLayout *layout = new QVBoxLayout(ui->chartTab);
    layout->addWidget(m_chartWidget);
    layout->setContentsMargins(0,0,0,0);

    connect(ui->calculateButton, &QPushButton::clicked, this, &ModelWidget1::onCalculateClicked);
    connect(ui->resetButton, &QPushButton::clicked, this, &ModelWidget1::onResetParameters);
    connect(ui->exportButton, &QPushButton::clicked, this, &ModelWidget1::onExportResults);
    connect(ui->resetViewButton, &QPushButton::clicked, this, &ModelWidget1::onResetView);
    connect(ui->fitToDataButton, &QPushButton::clicked, this, &ModelWidget1::onFitToData);

    onResetParameters();
}

ModelWidget1::~ModelWidget1() { delete ui; }

void ModelWidget1::onResetParameters() {
    // 对应 Composite_shale_oil_reservoir_fitfun.m 中的默认值
    ui->kfSpinBox->setValue(1e-3);
    ui->kmSpinBox->setValue(1e-4);
    ui->LSpinBox->setValue(1000);
    ui->LfSpinBox->setValue(100);
    ui->LfDSpinBox->setValue(0.1);
    ui->rmDSpinBox->setValue(4.0);
    ui->omga1SpinBox->setValue(0.4);
    ui->omga2SpinBox->setValue(0.08);
    ui->remda1SpinBox->setValue(0.001);
    ui->cDSpinBox->setValue(0);
    ui->sSpinBox->setValue(0);
}

void ModelWidget1::onResetView() { m_chartWidget->resetView(); }
void ModelWidget1::onFitToData() { m_chartWidget->autoFitData(); }

QMap<QString, double> ModelWidget1::collectParameters() {
    QMap<QString, double> params;
    params["kf"] = ui->kfSpinBox->value();
    params["km"] = ui->kmSpinBox->value();
    params["L"] = ui->LSpinBox->value();
    params["Lf"] = ui->LfSpinBox->value();
    params["LfD"] = ui->LfDSpinBox->value();
    params["rmD"] = ui->rmDSpinBox->value();
    params["omega1"] = ui->omga1SpinBox->value();
    params["omega2"] = ui->omga2SpinBox->value();
    params["lambda1"] = ui->remda1SpinBox->value();
    params["cD"] = ui->cDSpinBox->value();
    params["S"] = ui->sSpinBox->value();
    // 精度控制
    params["N"] = 8.0;
    return params;
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

void ModelWidget1::runCalculation() {
    QMap<QString, double> params = collectParameters();

    // 1. 生成时间步长 (Logspace -3 到 3)
    QVector<double> t;
    int steps = 100;
    for (int k = 0; k < steps; ++k) {
        double exp = -3.0 + (3.0 - (-3.0)) * k / (steps - 1);
        t.append(pow(10.0, exp));
    }

    // 2. 调用 ModelManager 进行核心计算
    // InfiniteConductive 此时已被映射为 Composite Model
    ModelManager manager;
    ModelCurveData result = manager.calculateTheoreticalCurve(ModelManager::InfiniteConductive, params, t);

    res_tD = std::get<0>(result); // 这里实际上是 t (hours/days)
    res_pD = std::get<1>(result); // Dp (MPa)
    res_dpD = std::get<2>(result); // dDp (MPa)

    // 3. 绘图
    m_chartWidget->setData(res_tD, res_pD, res_dpD);

    QString resultText = QString("计算完成\n模型: 复合页岩油储层试井解释模型\n点数: %1\n").arg(steps);
    resultText += "t(Time)\t\tDp(MPa)\t\tdDp(MPa)\n";
    for(int i=0; i<20 && i<res_pD.size(); ++i) {
        resultText += QString("%1\t%2\t%3\n").arg(res_tD[i],0,'e',4).arg(res_pD[i],0,'e',4).arg(res_dpD[i],0,'e',4);
    }
    ui->resultTextEdit->setText(resultText);

    emit calculationCompleted("Composite_Shale_Oil", params);
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
