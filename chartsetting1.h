#ifndef CHARTSETTING1_H
#define CHARTSETTING1_H

#include <QDialog>
#include <QLineEdit>
#include <QSpinBox>
#include <QCheckBox>
#include "qcustomplot.h"

/**
 * @brief 图表设置对话框
 * 用于修改图表标题、坐标轴标签、字体大小、网格线等属性。
 */
class ChartSetting1 : public QDialog {
    Q_OBJECT
public:
    /**
     * @param plot 指向需要设置的绘图控件
     * @param titleElement 指向图表标题对象（如果不存在则需要先创建）
     */
    explicit ChartSetting1(QCustomPlot* plot, QCPTextElement* titleElement, QWidget* parent = nullptr);

private slots:
    void onPreview(); ///< 预览效果
    void onAccept();  ///< 确认并保存
    void onReject();  ///< 取消并回滚

private:
    void applySettingsToPlot();          ///< 应用设置
    void restoreStateToPlot();           ///< 恢复设置

    QCustomPlot* m_plot;
    QCPTextElement* m_titleElement;

    // UI 控件
    QLineEdit* m_editTitle;
    QSpinBox* m_spinTitleFont;
    QLineEdit* m_editXLabel;
    QLineEdit* m_editYLabel;
    QSpinBox* m_spinLabelFont;
    QSpinBox* m_spinTickFont;
    QCheckBox* m_chkScientific;
    QCheckBox* m_chkGrid;

    // 状态备份，用于取消时恢复
    struct State {
        QString titleText;
        int titleFontSize;
        QString xLabel, yLabel;
        int labelFontSize, tickFontSize;
        bool isScientific;
        bool isGridVisible;
    } m_initialState;
};

#endif // CHARTSETTING1_H
