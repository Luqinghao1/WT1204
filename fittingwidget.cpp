#include "fittingwidget.h"
#include "ui_fittingwidget.h"
#include "pressurederivativecalculator.h"
#include <QtConcurrent>
#include <QMessageBox>
#include <QDebug>
#include <cmath>
#include <QFileDialog>
#include <QFile>
#include <QTextStream>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QPushButton>
#include <QLabel>
#include <Eigen/Dense>

// ===========================================================================
// FittingDataLoadDialog Implementation (数据加载对话框实现)
// ===========================================================================
FittingDataLoadDialog::FittingDataLoadDialog(const QList<QStringList>& previewData, QWidget *parent) : QDialog(parent) {
    setWindowTitle("数据列映射配置"); resize(800, 550);
    setStyleSheet("QDialog { background-color: #f0f0f0; } QLabel, QComboBox, QPushButton, QTableWidget, QGroupBox { color: black; }");
    QVBoxLayout* layout = new QVBoxLayout(this);

    layout->addWidget(new QLabel("请指定数据列含义 (时间必选):", this));

    // 数据预览表格
    m_previewTable = new QTableWidget(this);
    if(!previewData.isEmpty()) {
        int rows = qMin(previewData.size(), 50); int cols = previewData[0].size();
        m_previewTable->setRowCount(rows); m_previewTable->setColumnCount(cols);
        QStringList headers; for(int i=0;i<cols;++i) headers<<QString("Col %1").arg(i+1);
        m_previewTable->setHorizontalHeaderLabels(headers);
        for(int i=0;i<rows;++i) for(int j=0;j<cols && j<previewData[i].size();++j)
                m_previewTable->setItem(i,j,new QTableWidgetItem(previewData[i][j]));
    }
    m_previewTable->setAlternatingRowColors(true); layout->addWidget(m_previewTable);

    // 设置区域
    QGroupBox* grp = new QGroupBox("列映射与设置", this); QGridLayout* grid = new QGridLayout(grp);
    QStringList opts; for(int i=0;i<m_previewTable->columnCount();++i) opts<<QString("Col %1").arg(i+1);

    grid->addWidget(new QLabel("时间列 *:",this), 0, 0); m_comboTime = new QComboBox(this); m_comboTime->addItems(opts); grid->addWidget(m_comboTime, 0, 1);

    grid->addWidget(new QLabel("压力列:",this), 0, 2); m_comboPressure = new QComboBox(this); m_comboPressure->addItem("不导入",-1); m_comboPressure->addItems(opts); if(opts.size()>1) m_comboPressure->setCurrentIndex(2); grid->addWidget(m_comboPressure, 0, 3);

    grid->addWidget(new QLabel("导数列:",this), 1, 0); m_comboDeriv = new QComboBox(this); m_comboDeriv->addItem("自动计算 (Bourdet)",-1); m_comboDeriv->addItems(opts); grid->addWidget(m_comboDeriv, 1, 1);

    grid->addWidget(new QLabel("跳过首行数:",this), 1, 2); m_comboSkipRows = new QComboBox(this); for(int i=0;i<=20;++i) m_comboSkipRows->addItem(QString::number(i),i); m_comboSkipRows->setCurrentIndex(1); grid->addWidget(m_comboSkipRows, 1, 3);

    grid->addWidget(new QLabel("压力数据类型:",this), 2, 0); m_comboPressureType = new QComboBox(this); m_comboPressureType->addItem("原始压力 (自动计算压差 |P-Pi|)", 0); m_comboPressureType->addItem("压差数据 (直接使用 ΔP)", 1); grid->addWidget(m_comboPressureType, 2, 1, 1, 3);

    layout->addWidget(grp);

    // 按钮
    QHBoxLayout* btns = new QHBoxLayout; QPushButton* ok = new QPushButton("确定",this); QPushButton* cancel = new QPushButton("取消",this);
    connect(ok, &QPushButton::clicked, this, &FittingDataLoadDialog::validateSelection); connect(cancel, &QPushButton::clicked, this, &QDialog::reject);
    btns->addStretch(); btns->addWidget(ok); btns->addWidget(cancel); layout->addLayout(btns);
}
void FittingDataLoadDialog::validateSelection() { if(m_comboTime->currentIndex()<0) return; accept(); }
int FittingDataLoadDialog::getTimeColumnIndex() const { return m_comboTime->currentIndex(); }
int FittingDataLoadDialog::getPressureColumnIndex() const { return m_comboPressure->currentIndex()-1; }
int FittingDataLoadDialog::getDerivativeColumnIndex() const { return m_comboDeriv->currentIndex()-1; }
int FittingDataLoadDialog::getSkipRows() const { return m_comboSkipRows->currentData().toInt(); }
int FittingDataLoadDialog::getPressureDataType() const { return m_comboPressureType->currentData().toInt(); }


// ===========================================================================
// FittingWidget Implementation (拟合主界面实现)
// ===========================================================================

FittingWidget::FittingWidget(QWidget *parent) : QWidget(parent), ui(new Ui::FittingWidget), m_modelManager(nullptr), m_plotTitle(nullptr), m_isFitting(false)
{
    ui->setupUi(this);
    // 设置全局样式
    this->setStyleSheet("QWidget { color: black; font-family: Arial; } "
                        "QGroupBox { font-weight: bold; border: 1px solid gray; margin-top: 10px; } "
                        "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }");

    // 初始化分割器比例
    ui->splitter->setSizes(QList<int>{420, 680});
    ui->splitter->setCollapsible(0, false);
    ui->tableParams->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

    // 初始化绘图控件 (使用 MouseZoom)
    m_plot = new MouseZoom(this);
    ui->plotContainer->layout()->addWidget(m_plot);
    setupPlot();

    // 注册元类型，用于跨线程信号传递
    qRegisterMetaType<QMap<QString,double>>("QMap<QString,double>");
    qRegisterMetaType<ModelManager::ModelType>("ModelManager::ModelType");
    qRegisterMetaType<QVector<double>>("QVector<double>");

    // 信号连接
    connect(this, &FittingWidget::sigIterationUpdated, this, &FittingWidget::onIterationUpdate, Qt::QueuedConnection);
    connect(this, &FittingWidget::sigProgress, ui->progressBar, &QProgressBar::setValue);
    connect(&m_watcher, &QFutureWatcher<void>::finished, this, &FittingWidget::onFitFinished);

    // 权重调节逻辑：滑动条与数值框联动
    connect(ui->sliderWeight, &QSlider::valueChanged, this, [this](int val){
        ui->spinWeight->blockSignals(true);
        ui->spinWeight->setValue(val / 100.0);
        ui->spinWeight->blockSignals(false);
    });
    connect(ui->spinWeight, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double val){
        ui->sliderWeight->blockSignals(true);
        ui->sliderWeight->setValue((int)(val * 100));
        ui->sliderWeight->blockSignals(false);
    });
}

FittingWidget::~FittingWidget() { delete ui; }

void FittingWidget::setModelManager(ModelManager *m) {
    m_modelManager = m;
    // 如果管理器中已有数据，自动加载
    if (m_modelManager && m_modelManager->hasObservedData()) {
        QVector<double> t, p, d;
        m_modelManager->getObservedData(t, p, d);
        setObservedData(t, p, d);
    }
    initModelCombo();
}

void FittingWidget::initModelCombo() {
    if(!m_modelManager) return;
    ui->comboModelSelect->clear();
    ui->comboModelSelect->addItems(ModelManager::getAvailableModelTypes());
    ui->comboModelSelect->setCurrentIndex((int)m_modelManager->getCurrentModelType());
    on_btnResetParams_clicked();
}

/**
 * @brief 初始化图表样式
 */
void FittingWidget::setupPlot() {
    // 基础交互配置
    m_plot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);
    m_plot->setBackground(Qt::white); m_plot->axisRect()->setBackground(Qt::white);

    // 1. 设置默认标题
    m_plot->plotLayout()->insertRow(0);
    m_plotTitle = new QCPTextElement(m_plot, "试井解释", QFont("SimHei", 14, QFont::Bold));
    m_plot->plotLayout()->addElement(0, 0, m_plotTitle);

    // 2. 配置双对数坐标
    QSharedPointer<QCPAxisTickerLog> logTicker(new QCPAxisTickerLog);
    m_plot->xAxis->setScaleType(QCPAxis::stLogarithmic); m_plot->xAxis->setTicker(logTicker);
    m_plot->yAxis->setScaleType(QCPAxis::stLogarithmic); m_plot->yAxis->setTicker(logTicker);

    // 3. 科学计数法格式
    m_plot->xAxis->setNumberFormat("eb"); m_plot->xAxis->setNumberPrecision(0);
    m_plot->yAxis->setNumberFormat("eb"); m_plot->yAxis->setNumberPrecision(0);

    // 4. 字体与标签
    QFont labelFont("Arial", 12, QFont::Bold); QFont tickFont("Arial", 12);
    m_plot->xAxis->setLabel("时间 Time (h)"); m_plot->yAxis->setLabel("压力 & 导数 Pressure & Derivative (MPa)");
    m_plot->xAxis->setLabelFont(labelFont); m_plot->yAxis->setLabelFont(labelFont);
    m_plot->xAxis->setTickLabelFont(tickFont); m_plot->yAxis->setTickLabelFont(tickFont);

    // 5. 封闭坐标系 (上下左右都有轴线)
    m_plot->xAxis2->setVisible(true); m_plot->yAxis2->setVisible(true);
    m_plot->xAxis2->setTickLabels(false); m_plot->yAxis2->setTickLabels(false);
    connect(m_plot->xAxis, SIGNAL(rangeChanged(QCPRange)), m_plot->xAxis2, SLOT(setRange(QCPRange)));
    connect(m_plot->yAxis, SIGNAL(rangeChanged(QCPRange)), m_plot->yAxis2, SLOT(setRange(QCPRange)));
    m_plot->xAxis2->setScaleType(QCPAxis::stLogarithmic); m_plot->yAxis2->setScaleType(QCPAxis::stLogarithmic);
    m_plot->xAxis2->setTicker(logTicker); m_plot->yAxis2->setTicker(logTicker);

    // 6. 网格线
    m_plot->xAxis->grid()->setVisible(true); m_plot->yAxis->grid()->setVisible(true);
    m_plot->xAxis->grid()->setSubGridVisible(true); m_plot->yAxis->grid()->setSubGridVisible(true);
    m_plot->xAxis->grid()->setPen(QPen(QColor(220, 220, 220), 1, Qt::SolidLine));
    m_plot->yAxis->grid()->setPen(QPen(QColor(220, 220, 220), 1, Qt::SolidLine));
    m_plot->xAxis->grid()->setSubGridPen(QPen(QColor(240, 240, 240), 1, Qt::DotLine));
    m_plot->yAxis->grid()->setSubGridPen(QPen(QColor(240, 240, 240), 1, Qt::DotLine));

    m_plot->xAxis->setRange(1e-3, 1e3); m_plot->yAxis->setRange(1e-3, 1e2);

    // 7. 初始化曲线对象
    // Graph 0: 实测压力 (绿色圆点)
    m_plot->addGraph(); m_plot->graph(0)->setPen(Qt::NoPen);
    m_plot->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, QColor(0, 100, 0), 6));
    m_plot->graph(0)->setName("实测压力");

    // Graph 1: 实测导数 (品红三角)
    m_plot->addGraph(); m_plot->graph(1)->setPen(Qt::NoPen);
    m_plot->graph(1)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssTriangle, Qt::magenta, 6));
    m_plot->graph(1)->setName("实测导数");

    // Graph 2: 理论压力 (红色实线)
    m_plot->addGraph(); m_plot->graph(2)->setPen(QPen(Qt::red, 2));
    m_plot->graph(2)->setName("理论压力");

    // Graph 3: 理论导数 (蓝色实线)
    m_plot->addGraph(); m_plot->graph(3)->setPen(QPen(Qt::blue, 2));
    m_plot->graph(3)->setName("理论导数");

    m_plot->legend->setVisible(true); m_plot->legend->setFont(QFont("Arial", 9)); m_plot->legend->setBrush(QBrush(QColor(255, 255, 255, 200)));
}

/**
 * @brief 设置实测数据并绘图
 */
void FittingWidget::setObservedData(const QVector<double>& t, const QVector<double>& p, const QVector<double>& d) {
    m_obsTime = t; m_obsPressure = p; m_obsDerivative = d;
    if (m_modelManager) m_modelManager->setObservedData(t, p, d);

    // 过滤掉无效数据 (<=0)
    QVector<double> vt, vp, vd;
    for(int i=0; i<t.size(); ++i) {
        if(t[i]>1e-6 && p[i]>1e-6) {
            vt<<t[i]; vp<<p[i];
            if(i<d.size() && d[i]>1e-6) vd<<d[i]; else vd<<1e-10;
        }
    }

    m_plot->graph(0)->setData(vt, vp);
    m_plot->graph(1)->setData(vt, vd);
    m_plot->rescaleAxes();

    // 防止对数轴错误
    if(m_plot->xAxis->range().lower<=0) m_plot->xAxis->setRangeLower(1e-3);
    if(m_plot->yAxis->range().lower<=0) m_plot->yAxis->setRangeLower(1e-3);
    m_plot->replot();
}

void FittingWidget::on_comboModelSelect_currentIndexChanged(int) { on_btnResetParams_clicked(); }

void FittingWidget::on_btnResetView_clicked() {
    if(m_plot->graph(0)->dataCount() > 0) {
        m_plot->rescaleAxes();
        if(m_plot->xAxis->range().lower<=0) m_plot->xAxis->setRangeLower(1e-3);
        if(m_plot->yAxis->range().lower<=0) m_plot->yAxis->setRangeLower(1e-3);
    } else {
        m_plot->xAxis->setRange(1e-3, 1e3); m_plot->yAxis->setRange(1e-3, 1e2);
    }
    m_plot->replot();
}

/**
 * @brief 核心辅助函数：获取参数的显示信息
 * 统一处理 UI 显示（HTML）和 导出显示（Unicode）的上下标逻辑
 */
void FittingWidget::getParamDisplayInfo(const QString& key, QString& outName, QString& outSymbol, QString& outUnicodeSymbol, QString& outUnit) {
    QString unitMd = "mD";
    QString unitM = "m";
    QString unitDimless = "无因次";
    QString unitM3D = "m³/d";
    QString unitVis = "mPa·s";
    QString unitComp = "MPa⁻¹";
    QString unitVol = "";

    outName = key; outSymbol = key; outUnicodeSymbol = key; outUnit = "";

    if (key == "kf") { outName = "内区渗透率"; outSymbol = "k<sub>f</sub>"; outUnicodeSymbol = "k_f"; outUnit = unitMd; }
    else if (key == "km") { outName = "外区渗透率"; outSymbol = "k<sub>m</sub>"; outUnicodeSymbol = "kₘ"; outUnit = unitMd; }
    else if (key == "L") { outName = "水平井长度"; outSymbol = "L"; outUnicodeSymbol = "L"; outUnit = unitM; }
    else if (key == "Lf") { outName = "裂缝半长"; outSymbol = "L<sub>f</sub>"; outUnicodeSymbol = "L_f"; outUnit = unitM; }
    else if (key == "rmD") { outName = "复合半径"; outSymbol = "r<sub>mD</sub>"; outUnicodeSymbol = "rₘᴅ"; outUnit = unitDimless; }
    else if (key == "omega1") { outName = "内区储容比"; outSymbol = "ω<sub>1</sub>"; outUnicodeSymbol = "ω₁"; outUnit = unitDimless; }
    else if (key == "omega2") { outName = "外区储容比"; outSymbol = "ω<sub>2</sub>"; outUnicodeSymbol = "ω₂"; outUnit = unitDimless; }
    else if (key == "lambda1") { outName = "窜流系数"; outSymbol = "λ<sub>1</sub>"; outUnicodeSymbol = "λ₁"; outUnit = unitDimless; }
    else if (key == "omega") { outName = "储容比"; outSymbol = "ω"; outUnicodeSymbol = "ω"; outUnit = unitDimless; }
    else if (key == "lambda") { outName = "窜流系数"; outSymbol = "λ"; outUnicodeSymbol = "λ"; outUnit = unitDimless; }
    else if (key == "cD") { outName = "井筒储存"; outSymbol = "C<sub>D</sub>"; outUnicodeSymbol = "Cᴅ"; outUnit = unitDimless; }
    else if (key == "S") { outName = "表皮系数"; outSymbol = "S"; outUnicodeSymbol = "S"; outUnit = unitDimless; }
    else if (key == "phi") { outName = "孔隙度"; outSymbol = "φ"; outUnicodeSymbol = "φ"; outUnit = "小数"; }
    else if (key == "h") { outName = "厚度"; outSymbol = "h"; outUnicodeSymbol = "h"; outUnit = unitM; }
    else if (key == "mu") { outName = "粘度"; outSymbol = "μ"; outUnicodeSymbol = "μ"; outUnit = unitVis; }
    else if (key == "B") { outName = "体积系数"; outSymbol = "B"; outUnicodeSymbol = "B"; outUnit = unitVol; }
    else if (key == "Ct") { outName = "综合压缩系数"; outSymbol = "C<sub>t</sub>"; outUnicodeSymbol = "Cₜ"; outUnit = unitComp; }
    else if (key == "q") { outName = "产量"; outSymbol = "q"; outUnicodeSymbol = "q"; outUnit = unitM3D; }
    else if (key == "nf") { outName = "裂缝条数"; outSymbol = "n<sub>f</sub>"; outUnicodeSymbol = "n_f"; outUnit = unitDimless; }
}

QStringList FittingWidget::getParamOrder(ModelManager::ModelType type) {
    QStringList order;
    order << "phi" << "h" << "mu" << "B" << "Ct" << "q" << "nf";
    if (type == ModelManager::InfiniteConductive) {
        order << "kf" << "km" << "L" << "Lf" << "rmD" << "omega1" << "omega2" << "lambda1" << "cD" << "S";
    } else {
        order << "omega" << "lambda" << "cD" << "S";
    }
    return order;
}

/**
 * @brief 重置参数为默认值
 */
void FittingWidget::on_btnResetParams_clicked() {
    if(!m_modelManager) return;
    ModelManager::ModelType type = (ModelManager::ModelType)ui->comboModelSelect->currentIndex();
    QMap<QString,double> defs = m_modelManager->getDefaultParameters(type);

    // 默认兜底值
    if(!defs.contains("phi")) defs["phi"] = 0.05;
    if(!defs.contains("h")) defs["h"] = 20.0;
    if(!defs.contains("mu")) defs["mu"] = 0.5;
    if(!defs.contains("B")) defs["B"] = 1.05;
    if(!defs.contains("Ct")) defs["Ct"] = 5e-4;
    if(!defs.contains("q")) defs["q"] = 5.0;
    if(!defs.contains("nf")) defs["nf"] = 4.0;

    m_parameters.clear();
    QStringList orderedKeys = getParamOrder(type);

    for(const QString& key : orderedKeys) {
        if(defs.contains(key)) {
            FitParameter p;
            p.name = key;
            QString dummy1, dummy2, dummy3;
            getParamDisplayInfo(key, p.displayName, p.symbol, dummy2, p.unit);
            p.value = defs[key];
            p.isFit = false; // 默认不拟合

            // 设置拟合上下限经验值
            if (key == "kf" || key == "km") { p.min = 1e-6; p.max = 100.0; }
            else if (key == "L") { p.min = 10.0; p.max = 5000.0; }
            else if (key == "Lf") { p.min = 1.0; p.max = 1000.0; }
            else if (key == "rmD") { p.min = 1.0; p.max = 50.0; }
            else if (key == "omega1" || key == "omega2") { p.min = 0.001; p.max = 1.0; }
            else if (key == "lambda1") { p.min = 1e-9; p.max = 1.0; }
            else if (key == "cD") { p.min = 0.0; p.max = 100.0; }
            else if (key == "S") { p.min = 0; p.max = 50.0; }
            else if (key == "phi") { p.min = 0.001; p.max = 1.0; }
            else if (key == "h") { p.min = 1.0; p.max = 500.0; }
            else if (key == "mu") { p.min = 0.01; p.max = 1000.0; }
            else if (key == "B") { p.min = 0.5; p.max = 2.0; }
            else if (key == "Ct") { p.min = 1e-6; p.max = 1e-2; }
            else if (key == "q") { p.min = 0.1; p.max = 10000.0; }
            else if (key == "nf") { p.min = 1.0; p.max = 100.0; }
            else {
                if(p.value > 0) { p.min = p.value * 0.001; p.max = p.value * 1000.0; }
                else if (p.value == 0) { p.min = 0.0; p.max = 100.0; }
                else { p.min = -100.0; p.max = 100.0; }
            }
            m_parameters.append(p);
        }
    }
    loadParamsToTable();
    // 清除旧的理论曲线
    m_plot->graph(2)->data()->clear(); m_plot->graph(3)->data()->clear(); m_plot->replot();
}

/**
 * @brief 将内存参数加载到表格显示
 */
void FittingWidget::loadParamsToTable() {
    ui->tableParams->setRowCount(0);
    ui->tableParams->blockSignals(true);
    for(int i=0; i<m_parameters.size(); ++i) {
        ui->tableParams->insertRow(i);

        // 1. 参数名列 (富文本渲染)
        QString htmlSym, uniSym, unitStr, dummyName;
        getParamDisplayInfo(m_parameters[i].name, dummyName, htmlSym, uniSym, unitStr);
        QString nameStr = QString("<html>%1 (%2)</html>").arg(m_parameters[i].displayName).arg(htmlSym);

        QLabel* nameLabel = new QLabel(nameStr);
        nameLabel->setTextFormat(Qt::RichText);
        nameLabel->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        nameLabel->setContentsMargins(5, 0, 0, 0);
        ui->tableParams->setCellWidget(i, 0, nameLabel);

        // 隐藏 Key
        QTableWidgetItem* dummyItem = new QTableWidgetItem("");
        dummyItem->setData(Qt::UserRole, m_parameters[i].name);
        ui->tableParams->setItem(i, 0, dummyItem);

        // 2. 拟合值
        ui->tableParams->setItem(i, 1, new QTableWidgetItem(QString::number(m_parameters[i].value)));

        // 3. 拟合勾选
        QTableWidgetItem* chk = new QTableWidgetItem();
        chk->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled | Qt::ItemIsSelectable);
        chk->setCheckState(m_parameters[i].isFit ? Qt::Checked : Qt::Unchecked);
        ui->tableParams->setItem(i, 2, chk);

        // 4 & 5. 上下限
        ui->tableParams->setItem(i, 3, new QTableWidgetItem(QString::number(m_parameters[i].min)));
        ui->tableParams->setItem(i, 4, new QTableWidgetItem(QString::number(m_parameters[i].max)));

        // 6. 单位
        if(unitStr == "无因次" || unitStr == "小数") unitStr = "";
        QTableWidgetItem* unitItem = new QTableWidgetItem(unitStr);
        unitItem->setFlags(unitItem->flags() ^ Qt::ItemIsEditable); // 只读
        ui->tableParams->setItem(i, 5, unitItem);
    }
    ui->tableParams->blockSignals(false);
}

void FittingWidget::updateParamsFromTable() {
    for(int i=0; i<ui->tableParams->rowCount(); ++i) {
        if(i < m_parameters.size()) {
            QString key = ui->tableParams->item(i,0)->data(Qt::UserRole).toString();
            if(m_parameters[i].name == key) {
                m_parameters[i].value = ui->tableParams->item(i,1)->text().toDouble();
                m_parameters[i].isFit = (ui->tableParams->item(i,2)->checkState() == Qt::Checked);
                m_parameters[i].min = ui->tableParams->item(i,3)->text().toDouble();
                m_parameters[i].max = ui->tableParams->item(i,4)->text().toDouble();
            }
        }
    }
}

QStringList FittingWidget::parseLine(const QString& line) { return line.split(QRegularExpression("[,\\s\\t]+"), Qt::SkipEmptyParts); }

void FittingWidget::on_btnLoadData_clicked() {
    QString path = QFileDialog::getOpenFileName(this, "加载试井数据", "", "文本文件 (*.txt *.csv)");
    if(path.isEmpty()) return;
    QFile f(path); if(!f.open(QIODevice::ReadOnly)) return;
    QTextStream in(&f); QList<QStringList> data;
    while(!in.atEnd()) { QString l=in.readLine().trimmed(); if(!l.isEmpty()) data<<parseLine(l); }
    f.close();

    FittingDataLoadDialog dlg(data, this);
    if(dlg.exec()!=QDialog::Accepted) return;

    int tCol=dlg.getTimeColumnIndex(), pCol=dlg.getPressureColumnIndex(), dCol=dlg.getDerivativeColumnIndex();
    int pressureType = dlg.getPressureDataType();
    QVector<double> t, p, d;
    double p_init = 0;

    // 如果选择的是原始压力，尝试获取初始压力
    if(pressureType == 0 && pCol>=0) {
        for(int i=dlg.getSkipRows(); i<data.size(); ++i) {
            if(pCol<data[i].size()) { p_init = data[i][pCol].toDouble(); break; }
        }
    }

    for(int i=dlg.getSkipRows(); i<data.size(); ++i) {
        if(tCol<data[i].size()) {
            double tv = data[i][tCol].toDouble();
            double pv = 0;
            if (pCol>=0 && pCol<data[i].size()) {
                double val = data[i][pCol].toDouble();
                pv = (pressureType == 0) ? std::abs(val - p_init) : val;
            }
            if(tv>0) { t<<tv; p<<pv; }
        }
    }

    if (dCol >= 0) {
        for(int i=dlg.getSkipRows(); i<data.size(); ++i)
            if(tCol<data[i].size() && data[i][tCol].toDouble() > 0 && dCol<data[i].size()) d << data[i][dCol].toDouble();
    } else {
        d = PressureDerivativeCalculator::calculateBourdetDerivative(t, p, 0.15);
    }
    setObservedData(t, p, d);
}

void FittingWidget::on_btnRunFit_clicked() {
    if(m_isFitting) return;
    if(m_obsTime.isEmpty()) { QMessageBox::warning(this,"错误","请先加载观测数据。"); return; }

    updateParamsFromTable();
    m_isFitting = true; m_stopRequested = false; ui->btnRunFit->setEnabled(false);

    ModelManager::ModelType modelType = (ModelManager::ModelType)ui->comboModelSelect->currentIndex();
    QList<FitParameter> paramsCopy = m_parameters;
    double w = ui->spinWeight->value();

    // 启动后台线程
    (void)QtConcurrent::run([this, modelType, paramsCopy, w](){ runOptimizationTask(modelType, paramsCopy, w); });
}

void FittingWidget::runOptimizationTask(ModelManager::ModelType modelType, QList<FitParameter> fitParams, double weight) {
    runLevenbergMarquardtOptimization(modelType, fitParams, weight);
}

void FittingWidget::on_btnStop_clicked() { m_stopRequested=true; }
void FittingWidget::on_btnImportModel_clicked() { updateModelCurve(); }

void FittingWidget::on_btnExportData_clicked() {
    updateParamsFromTable();
    QString fileName = QFileDialog::getSaveFileName(this, "导出拟合参数", "FittingParameters.csv",
                                                    "CSV Files (*.csv);;Text Files (*.txt)");
    if (fileName.isEmpty()) return;

    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::critical(this, "错误", "无法写入文件。");
        return;
    }

    QTextStream out(&file);
    if(fileName.endsWith(".csv", Qt::CaseInsensitive)) {
        file.write("\xEF\xBB\xBF"); // BOM
        out << QString("参数中文名,参数英文名,拟合值,单位\n");
        for(const auto& param : m_parameters) {
            QString htmlSym, uniSym, unitStr, dummyName;
            getParamDisplayInfo(param.name, dummyName, htmlSym, uniSym, unitStr);
            if(unitStr == "无因次" || unitStr == "小数") unitStr = "";
            out << QString("%1,%2,%3,%4\n").arg(param.displayName).arg(uniSym).arg(param.value, 0, 'g', 10).arg(unitStr);
        }
    } else {
        for(const auto& param : m_parameters) {
            QString htmlSym, uniSym, unitStr, dummyName;
            getParamDisplayInfo(param.name, dummyName, htmlSym, uniSym, unitStr);
            if(unitStr == "无因次" || unitStr == "小数") unitStr = "";
            QString lineStr = QString("%1 (%2): %3 %4").arg(param.displayName).arg(uniSym).arg(param.value, 0, 'g', 10).arg(unitStr);
            out << lineStr.trimmed() << "\n";
        }
    }
    file.close();
    QMessageBox::information(this, "完成", "参数数据已成功导出。");
}

void FittingWidget::on_btnExportChart_clicked() {
    QString fileName = QFileDialog::getSaveFileName(this, "导出图表", "FittingChart.png",
                                                    "PNG Image (*.png);;JPEG Image (*.jpg);;PDF Document (*.pdf)");
    if (fileName.isEmpty()) return;
    bool success = false;
    if (fileName.endsWith(".png", Qt::CaseInsensitive)) success = m_plot->savePng(fileName);
    else if (fileName.endsWith(".jpg", Qt::CaseInsensitive)) success = m_plot->saveJpg(fileName);
    else if (fileName.endsWith(".pdf", Qt::CaseInsensitive)) success = m_plot->savePdf(fileName);
    else success = m_plot->savePng(fileName + ".png");

    if (success) QMessageBox::information(this, "完成", "图表已成功导出。");
    else QMessageBox::critical(this, "错误", "导出图表失败。");
}

void FittingWidget::on_btnChartSettings_clicked() {
    // 调用通用的图表设置弹窗 (ChartSetting1)
    ChartSetting1 dlg(m_plot, m_plotTitle, this);
    dlg.exec();
}

void FittingWidget::updateModelCurve() {
    if(!m_modelManager) { QMessageBox::critical(this, "错误", "ModelManager 未初始化！"); return; }
    ui->tableParams->clearFocus();
    updateParamsFromTable();

    QMap<QString,double> currentParams;
    for(const auto& p : m_parameters) currentParams.insert(p.name, p.value);

    // 联动更新 LfD
    if(currentParams.contains("L") && currentParams.contains("Lf") && currentParams["L"] > 1e-9)
        currentParams["LfD"] = currentParams["Lf"] / currentParams["L"];
    else
        currentParams["LfD"] = 0.0;

    ModelManager::ModelType type = (ModelManager::ModelType)ui->comboModelSelect->currentIndex();
    QVector<double> targetT = m_obsTime;
    if(targetT.isEmpty()) { for(double e = -4; e <= 4; e += 0.1) targetT.append(pow(10, e)); }

    ModelCurveData res = m_modelManager->calculateTheoreticalCurve(type, currentParams, targetT);
    onIterationUpdate(0, currentParams, std::get<0>(res), std::get<1>(res), std::get<2>(res));
}

// --------------------------------------------------------------------------------
// 拟合算法核心 (LM)
// --------------------------------------------------------------------------------
void FittingWidget::runLevenbergMarquardtOptimization(ModelManager::ModelType modelType, QList<FitParameter> params, double weight) {
    if(m_modelManager) m_modelManager->setHighPrecision(false); // 降低精度以提高速度

    QVector<int> fitIndices;
    for(int i=0; i<params.size(); ++i) if(params[i].isFit) fitIndices.append(i);
    int nParams = fitIndices.size();
    if(nParams == 0) { QMetaObject::invokeMethod(this, "onFitFinished"); return; }

    double lambda = 0.01;
    int maxIter = 50;
    double currentSSE = 1e15;

    QMap<QString, double> currentParamMap;
    for(const auto& p : params) currentParamMap.insert(p.name, p.value);

    // 初始化 LfD
    if(currentParamMap.contains("L") && currentParamMap.contains("Lf") && currentParamMap["L"] > 1e-9)
        currentParamMap["LfD"] = currentParamMap["Lf"] / currentParamMap["L"];

    QVector<double> residuals = calculateResiduals(currentParamMap, modelType, weight);
    currentSSE = calculateSumSquaredError(residuals);

    // 初始状态更新
    ModelCurveData curve = m_modelManager->calculateTheoreticalCurve(modelType, currentParamMap);
    emit sigIterationUpdated(currentSSE/residuals.size(), currentParamMap, std::get<0>(curve), std::get<1>(curve), std::get<2>(curve));

    for(int iter = 0; iter < maxIter; ++iter) {
        if(m_stopRequested) break;
        emit sigProgress(iter * 100 / maxIter);

        QVector<QVector<double>> J = computeJacobian(currentParamMap, residuals, fitIndices, modelType, params, weight);
        int nRes = residuals.size();

        QVector<QVector<double>> H(nParams, QVector<double>(nParams, 0.0));
        QVector<double> g(nParams, 0.0);

        for(int k=0; k<nRes; ++k) {
            for(int i=0; i<nParams; ++i) {
                g[i] += J[k][i] * residuals[k];
                for(int j=0; j<=i; ++j) H[i][j] += J[k][i] * J[k][j];
            }
        }
        for(int i=0; i<nParams; ++i) for(int j=i+1; j<nParams; ++j) H[i][j] = H[j][i];

        bool stepAccepted = false;
        for(int tryIter=0; tryIter<5; ++tryIter) {
            QVector<QVector<double>> H_lm = H;
            for(int i=0; i<nParams; ++i) H_lm[i][i] += lambda * (1.0 + std::abs(H[i][i]));

            QVector<double> negG(nParams); for(int i=0;i<nParams;++i) negG[i] = -g[i];
            QVector<double> delta = solveLinearSystem(H_lm, negG);

            QMap<QString, double> trialMap = currentParamMap;

            for(int i=0; i<nParams; ++i) {
                int pIdx = fitIndices[i];
                QString pName = params[pIdx].name;
                double oldVal = currentParamMap[pName];
                // 对数域更新判断 (除了 S 和 nf 外，其他大部分为正数)
                bool isLog = (oldVal > 1e-12 && pName != "S" && pName != "nf");

                double newVal;
                if(isLog) {
                    double logVal = log10(oldVal) + delta[i];
                    newVal = pow(10.0, logVal);
                } else {
                    newVal = oldVal + delta[i];
                }
                newVal = qMax(params[pIdx].min, qMin(newVal, params[pIdx].max));
                trialMap[pName] = newVal;
            }

            if(trialMap.contains("L") && trialMap.contains("Lf") && trialMap["L"] > 1e-9)
                trialMap["LfD"] = trialMap["Lf"] / trialMap["L"];

            QVector<double> newRes = calculateResiduals(trialMap, modelType, weight);
            double newSSE = calculateSumSquaredError(newRes);

            if(newSSE < currentSSE) {
                currentSSE = newSSE;
                currentParamMap = trialMap;
                residuals = newRes;
                lambda /= 10.0;
                stepAccepted = true;
                ModelCurveData iterCurve = m_modelManager->calculateTheoreticalCurve(modelType, currentParamMap);
                emit sigIterationUpdated(currentSSE/nRes, currentParamMap, std::get<0>(iterCurve), std::get<1>(iterCurve), std::get<2>(iterCurve));
                break;
            } else {
                lambda *= 10.0;
            }
        }
        if(!stepAccepted && lambda > 1e10) break;
    }

    if(m_modelManager) m_modelManager->setHighPrecision(true);
    // 最后更新一次高精度的
    if(currentParamMap.contains("L") && currentParamMap.contains("Lf") && currentParamMap["L"] > 1e-9)
        currentParamMap["LfD"] = currentParamMap["Lf"] / currentParamMap["L"];

    ModelCurveData finalCurve = m_modelManager->calculateTheoreticalCurve(modelType, currentParamMap);
    emit sigIterationUpdated(currentSSE/residuals.size(), currentParamMap, std::get<0>(finalCurve), std::get<1>(finalCurve), std::get<2>(finalCurve));
    QMetaObject::invokeMethod(this, "onFitFinished");
}

QVector<double> FittingWidget::calculateResiduals(const QMap<QString, double>& params, ModelManager::ModelType modelType, double weight) {
    if(!m_modelManager || m_obsTime.isEmpty()) return QVector<double>();
    ModelCurveData res = m_modelManager->calculateTheoreticalCurve(modelType, params, m_obsTime);
    const QVector<double>& pCal = std::get<1>(res);
    const QVector<double>& dpCal = std::get<2>(res);
    QVector<double> r;
    double wp = weight; double wd = 1.0 - weight;
    int count = qMin(m_obsPressure.size(), pCal.size());
    for(int i=0; i<count; ++i) {
        if(m_obsPressure[i] > 1e-10 && pCal[i] > 1e-10)
            r.append( (log(m_obsPressure[i]) - log(pCal[i])) * wp );
        else r.append(0.0);
    }
    int dCount = qMin(m_obsDerivative.size(), dpCal.size()); dCount = qMin(dCount, count);
    for(int i=0; i<dCount; ++i) {
        if(m_obsDerivative[i] > 1e-10 && dpCal[i] > 1e-10)
            r.append( (log(m_obsDerivative[i]) - log(dpCal[i])) * wd );
        else r.append(0.0);
    }
    return r;
}

QVector<QVector<double>> FittingWidget::computeJacobian(const QMap<QString, double>& params, const QVector<double>& baseResiduals, const QVector<int>& fitIndices, ModelManager::ModelType modelType, const QList<FitParameter>& currentFitParams, double weight) {
    int nRes = baseResiduals.size(); int nParams = fitIndices.size();
    QVector<QVector<double>> J(nRes, QVector<double>(nParams));
    for(int j = 0; j < nParams; ++j) {
        int idx = fitIndices[j]; QString pName = currentFitParams[idx].name;
        double val = params.value(pName); bool isLog = (val > 1e-12 && pName != "S" && pName != "nf");
        double h; QMap<QString, double> pPlus = params; QMap<QString, double> pMinus = params;
        if(isLog) { h = 0.01; double valLog = log10(val); pPlus[pName] = pow(10.0, valLog + h); pMinus[pName] = pow(10.0, valLog - h); }
        else { h = 1e-4; pPlus[pName] = val + h; pMinus[pName] = val - h; }
        auto updateDeps = [](QMap<QString,double>& map) { if(map.contains("L") && map.contains("Lf") && map["L"] > 1e-9) map["LfD"] = map["Lf"] / map["L"]; };
        if(pName == "L" || pName == "Lf") { updateDeps(pPlus); updateDeps(pMinus); }
        QVector<double> rPlus = calculateResiduals(pPlus, modelType, weight);
        QVector<double> rMinus = calculateResiduals(pMinus, modelType, weight);
        if(rPlus.size() == nRes && rMinus.size() == nRes) {
            for(int i=0; i<nRes; ++i) J[i][j] = (rPlus[i] - rMinus[i]) / (2.0 * h);
        }
    }
    return J;
}

QVector<double> FittingWidget::solveLinearSystem(const QVector<QVector<double>>& A, const QVector<double>& b) {
    int n = b.size(); if (n == 0) return QVector<double>();
    Eigen::MatrixXd matA(n, n); Eigen::VectorXd vecB(n);
    for (int i = 0; i < n; ++i) { vecB(i) = b[i]; for (int j = 0; j < n; ++j) matA(i, j) = A[i][j]; }
    Eigen::VectorXd x = matA.ldlt().solve(vecB);
    QVector<double> res(n); for (int i = 0; i < n; ++i) res[i] = x(i);
    return res;
}

double FittingWidget::calculateSumSquaredError(const QVector<double>& residuals) {
    double sse = 0.0; for(double v : residuals) sse += v*v; return sse;
}

void FittingWidget::onIterationUpdate(double err, const QMap<QString,double>& p,
                                      const QVector<double>& t, const QVector<double>& p_curve, const QVector<double>& d_curve) {
    ui->label_Error->setText(QString("误差(MSE): %1").arg(err, 0, 'e', 3));
    ui->tableParams->blockSignals(true);
    for(int i=0; i<ui->tableParams->rowCount(); ++i) {
        QString key = ui->tableParams->item(i, 0)->data(Qt::UserRole).toString();
        if(p.contains(key)) {
            double val = p[key];
            ui->tableParams->item(i, 1)->setText(QString::number(val, 'g', 5));
        }
    }
    ui->tableParams->blockSignals(false);
    plotCurves(t, p_curve, d_curve, true);
}

void FittingWidget::onFitFinished() { m_isFitting = false; ui->btnRunFit->setEnabled(true); QMessageBox::information(this, "完成", "拟合完成。"); }

void FittingWidget::plotCurves(const QVector<double>& t, const QVector<double>& p, const QVector<double>& d, bool isModel) {
    QVector<double> vt, vp, vd;
    for(int i=0; i<t.size(); ++i) {
        if(t[i]>1e-8 && p[i]>1e-8) {
            vt<<t[i]; vp<<p[i];
            if(i<d.size() && d[i]>1e-8) vd<<d[i]; else vd<<1e-10;
        }
    }
    if(isModel) {
        m_plot->graph(2)->setData(vt, vp); m_plot->graph(3)->setData(vt, vd);
        if (m_obsTime.isEmpty() && !vt.isEmpty()) {
            m_plot->rescaleAxes();
            if(m_plot->xAxis->range().lower<=0) m_plot->xAxis->setRangeLower(1e-3);
            if(m_plot->yAxis->range().lower<=0) m_plot->yAxis->setRangeLower(1e-3);
        }
        m_plot->replot();
    }
}
