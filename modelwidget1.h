#ifndef MODELWIDGET1_H
#define MODELWIDGET1_H

#include <QWidget>
#include <QMap>
#include <QVector>
#include <functional>
#include <tuple>
#include <QLineEdit>

// 引入新模块
#include "mousezoom.h"
#include "chartsetting1.h"

// 定义模型曲线数据类型: <时间(t), 压力(pD), 导数(dpD)>
typedef std::tuple<QVector<double>, QVector<double>, QVector<double>> ModelCurveData;

namespace Ui {
class ModelWidget1;
}

// ---------------------------------------------------------
// ModelWidget1 (业务逻辑 + 计算核心)
// ---------------------------------------------------------
/**
 * @brief 复合页岩油藏试井解释模型界面类
 * 包含参数输入(支持多值敏感性分析)、模型计算核心算法以及基于 QCustomPlot 的增强绘图。
 */
class ModelWidget1 : public QWidget
{
    Q_OBJECT

public:
    explicit ModelWidget1(QWidget *parent = nullptr);
    ~ModelWidget1();

    /**
     * @brief 计算理论曲线（对外接口）
     * @param params 模型参数字典 (key -> value)
     * @param providedTime 时间序列 (若为空则自动生成)
     * @return 元组 (Time, Pressure, Derivative)
     */
    ModelCurveData calculateTheoreticalCurve(const QMap<QString, double>& params,
                                             const QVector<double>& providedTime = QVector<double>());

    void setHighPrecision(bool high);

private slots:
    void onCalculateClicked();       ///< 点击“开始计算”按钮
    void onResetParameters();        ///< 点击“重置参数”按钮
    void onExportResults();          ///< 点击“导出结果”按钮
    void onResetView();              ///< 点击“重置视图”按钮
    void onFitToData();              ///< 点击“适应数据”按钮 (对应 QCustomPlot 的 Rescale)
    void onChartSettings();          ///< 点击“图表设置”按钮
    void onDependentParamsChanged(); ///< 联动更新 LfD

signals:
    void calculationCompleted(const QString &analysisType, const QMap<QString, double> &results);

private:
    // --- UI 初始化与辅助 ---
    void initChart();   ///< 初始化 QCustomPlot 图表
    void setupConnections();

    // --- 参数处理 ---
    /**
     * @brief 解析输入框内容
     * @param text 输入文本，支持科学计数法和逗号分隔 ("1e-3, 0.002")
     * @return 解析出的数值列表
     */
    QVector<double> parseInput(const QString& text);

    /**
     * @brief 设置输入框的显示文本
     * 去除多余的0，保留有效数字。
     */
    void setInputText(QLineEdit* edit, double value);

    /**
     * @brief 执行计算主流程
     * 处理单次计算和敏感性分析多重循环逻辑。
     */
    void runCalculation();

    // --- 绘图辅助 ---
    /**
     * @brief 绘制一组曲线
     * @param data 数据元组
     * @param name 图例名称
     * @param color 颜色
     * @param isSensitivity 是否为敏感性分析模式 (若是，则压力实线，导数虚线)
     */
    void plotCurve(const ModelCurveData& data, const QString& name, QColor color, bool isSensitivity);

    // --- 核心计算方法 (数值反演与数学模型) ---
    void calculatePDandDeriv(const QVector<double>& tD, const QMap<QString, double>& params,
                             std::function<double(double, const QMap<QString, double>&)> laplaceFunc,
                             QVector<double>& outPD, QVector<double>& outDeriv);

    double flaplace_composite(double z, const QMap<QString, double>& p);
    double PWD_inf(double z, double fs1, double fs2, double M12, double LfD, double rmD, int nf, const QVector<double>& xwD);
    double scaled_besseli(int v, double x);
    double adaptiveGauss(std::function<double(double)> f, double a, double b, double eps, int depth, int maxDepth);
    double gauss15(std::function<double(double)> f, double a, double b);
    double stefestCoefficient(int i, int N);
    double factorial(int n);

private:
    Ui::ModelWidget1 *ui;

    // 使用新的增强型绘图类
    MouseZoom *m_plot;
    QCPTextElement *m_plotTitle;

    bool m_highPrecision;

    // 缓存最后一次的主计算结果，用于导出
    QVector<double> res_tD;
    QVector<double> res_pD;
    QVector<double> res_dpD;

    // 敏感性分析颜色表
    QList<QColor> m_colorList;
};

#endif // MODELWIDGET1_H
