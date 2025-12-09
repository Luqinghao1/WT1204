#include "chartsetting1.h"
#include <QVBoxLayout>
#include <QFormLayout>
#include <QPushButton>

ChartSetting1::ChartSetting1(QCustomPlot* plot, QCPTextElement* titleElement, QWidget* parent)
    : QDialog(parent), m_plot(plot), m_titleElement(titleElement)
{
    setWindowTitle("图表显示设置");
    resize(400, 350);

    // 1. 备份初始状态
    if (m_titleElement) {
        m_initialState.titleText = m_titleElement->text();
        m_initialState.titleFontSize = m_titleElement->font().pointSize();
    } else {
        m_initialState.titleText = "";
        m_initialState.titleFontSize = 12;
    }
    m_initialState.xLabel = m_plot->xAxis->label();
    m_initialState.yLabel = m_plot->yAxis->label();
    m_initialState.labelFontSize = m_plot->xAxis->labelFont().pointSize();
    m_initialState.tickFontSize = m_plot->xAxis->tickLabelFont().pointSize();
    m_initialState.isScientific = (m_plot->xAxis->numberFormat().contains("e"));
    m_initialState.isGridVisible = m_plot->xAxis->grid()->visible();

    // 2. 构建界面
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    QFormLayout* form = new QFormLayout;

    m_editTitle = new QLineEdit(m_initialState.titleText, this);
    m_spinTitleFont = new QSpinBox(this); m_spinTitleFont->setRange(8, 72); m_spinTitleFont->setValue(m_initialState.titleFontSize);

    m_editXLabel = new QLineEdit(m_initialState.xLabel, this);
    m_editYLabel = new QLineEdit(m_initialState.yLabel, this);

    m_spinLabelFont = new QSpinBox(this); m_spinLabelFont->setRange(6, 48); m_spinLabelFont->setValue(m_initialState.labelFontSize);
    m_spinTickFont = new QSpinBox(this); m_spinTickFont->setRange(6, 48); m_spinTickFont->setValue(m_initialState.tickFontSize);

    m_chkScientific = new QCheckBox("启用科学计数法", this); m_chkScientific->setChecked(m_initialState.isScientific);
    m_chkGrid = new QCheckBox("显示网格线", this); m_chkGrid->setChecked(m_initialState.isGridVisible);

    form->addRow("图表标题:", m_editTitle);
    form->addRow("标题字号:", m_spinTitleFont);
    form->addRow("X轴 标题:", m_editXLabel);
    form->addRow("Y轴 标题:", m_editYLabel);
    form->addRow("轴标题字号:", m_spinLabelFont);
    form->addRow("刻度字号:", m_spinTickFont);
    form->addRow("数值格式:", m_chkScientific);
    form->addRow("网格线:", m_chkGrid);

    mainLayout->addLayout(form);

    QHBoxLayout* btnLayout = new QHBoxLayout;
    QPushButton* btnPreview = new QPushButton("预览", this);
    QPushButton* btnOk = new QPushButton("确定", this);
    QPushButton* btnCancel = new QPushButton("取消", this);
    btnOk->setDefault(true);
    btnLayout->addStretch();
    btnLayout->addWidget(btnPreview); btnLayout->addWidget(btnOk); btnLayout->addWidget(btnCancel);
    mainLayout->addLayout(btnLayout);

    connect(btnPreview, &QPushButton::clicked, this, &ChartSetting1::onPreview);
    connect(btnOk, &QPushButton::clicked, this, &ChartSetting1::onAccept);
    connect(btnCancel, &QPushButton::clicked, this, &ChartSetting1::onReject);
}

void ChartSetting1::onPreview() { applySettingsToPlot(); }
void ChartSetting1::onAccept() { applySettingsToPlot(); accept(); }
void ChartSetting1::onReject() { restoreStateToPlot(); reject(); }

void ChartSetting1::applySettingsToPlot() {
    if(!m_plot) return;

    // 应用标题
    if(m_titleElement) {
        m_titleElement->setText(m_editTitle->text());
        QFont f = m_titleElement->font();
        f.setPointSize(m_spinTitleFont->value());
        m_titleElement->setFont(f);
    }

    // 应用轴标签
    m_plot->xAxis->setLabel(m_editXLabel->text());
    m_plot->yAxis->setLabel(m_editYLabel->text());

    // 应用字体
    QFont lf = m_plot->xAxis->labelFont(); lf.setPointSize(m_spinLabelFont->value());
    m_plot->xAxis->setLabelFont(lf); m_plot->yAxis->setLabelFont(lf);

    QFont tf = m_plot->xAxis->tickLabelFont(); tf.setPointSize(m_spinTickFont->value());
    m_plot->xAxis->setTickLabelFont(tf); m_plot->yAxis->setTickLabelFont(tf);

    // 应用格式
    QString fmt = m_chkScientific->isChecked() ? "eb" : "g";
    m_plot->xAxis->setNumberFormat(fmt); m_plot->yAxis->setNumberFormat(fmt);
    // 确保上右轴同步
    m_plot->xAxis2->setNumberFormat(fmt); m_plot->yAxis2->setNumberFormat(fmt);

    // 应用网格
    m_plot->xAxis->grid()->setVisible(m_chkGrid->isChecked());
    m_plot->yAxis->grid()->setVisible(m_chkGrid->isChecked());

    m_plot->replot();
}

void ChartSetting1::restoreStateToPlot() {
    if(!m_plot) return;
    if(m_titleElement) {
        m_titleElement->setText(m_initialState.titleText);
        QFont f = m_titleElement->font(); f.setPointSize(m_initialState.titleFontSize);
        m_titleElement->setFont(f);
    }
    m_plot->xAxis->setLabel(m_initialState.xLabel);
    m_plot->yAxis->setLabel(m_initialState.yLabel);

    QFont lf = m_plot->xAxis->labelFont(); lf.setPointSize(m_initialState.labelFontSize);
    m_plot->xAxis->setLabelFont(lf); m_plot->yAxis->setLabelFont(lf);

    QFont tf = m_plot->xAxis->tickLabelFont(); tf.setPointSize(m_initialState.tickFontSize);
    m_plot->xAxis->setTickLabelFont(tf); m_plot->yAxis->setTickLabelFont(tf);

    QString fmt = m_initialState.isScientific ? "eb" : "g";
    m_plot->xAxis->setNumberFormat(fmt); m_plot->yAxis->setNumberFormat(fmt);
    m_plot->xAxis2->setNumberFormat(fmt); m_plot->yAxis2->setNumberFormat(fmt);

    m_plot->xAxis->grid()->setVisible(m_initialState.isGridVisible);
    m_plot->yAxis->grid()->setVisible(m_initialState.isGridVisible);

    m_plot->replot();
}
