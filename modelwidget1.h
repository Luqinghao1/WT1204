#ifndef MODELWIDGET1_H
#define MODELWIDGET1_H

#include <QWidget>
#include <QPainter>
#include <QVector>
#include <QMap>
#include <QMouseEvent>
#include <QWheelEvent>
#include <functional>

namespace Ui {
class ModelWidget1;
}

// ---------------------------------------------------------
// LogLogChartWidget (绘图组件，保持原样)
// ---------------------------------------------------------
class LogLogChartWidget : public QWidget
{
    Q_OBJECT
public:
    explicit LogLogChartWidget(QWidget *parent = nullptr);
    void setData(const QVector<double>& xData, const QVector<double>& yData1,
                 const QVector<double>& yData2, const QVector<double>& xData2 = QVector<double>());
    void clearData();
    void resetView();
    void autoFitData();
    void setShowOriginalData(bool show) { m_showOriginalData = show; update(); }
protected:
    void paintEvent(QPaintEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
private:
    void drawAxis(QPainter& painter, const QRect& plotRect);
    void drawData(QPainter& painter, const QRect& plotRect);
    void drawLegend(QPainter& painter, const QRect& plotRect);
    QPointF dataToPixel(double x, double y, const QRect& plotRect);
    QVector<double> m_xData, m_yData1, m_yData2, m_xData2;
    double m_xMin, m_xMax, m_yMin, m_yMax;
    bool m_hasData, m_showOriginalData, m_isDragging;
    QPoint m_lastMousePos;
};

// ---------------------------------------------------------
// ModelWidget1 (业务逻辑)
// ---------------------------------------------------------
class ModelWidget1 : public QWidget
{
    Q_OBJECT

public:
    explicit ModelWidget1(QWidget *parent = nullptr);
    ~ModelWidget1();

private slots:
    void onCalculateClicked();
    void onResetParameters();
    void onExportResults();
    void onResetView();
    void onFitToData();

signals:
    void calculationCompleted(const QString &analysisType, const QMap<QString, double> &results);

private:
    void runCalculation();
    // 收集UI参数
    QMap<QString, double> collectParameters();

private:
    Ui::ModelWidget1 *ui;
    LogLogChartWidget *m_chartWidget;

    // 结果数据缓存
    QVector<double> res_tD;
    QVector<double> res_pD;
    QVector<double> res_dpD;
    QVector<double> res_td_dpD;
};

#endif // MODELWIDGET1_H
