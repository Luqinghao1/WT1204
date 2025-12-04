#include "fittingwidget.h"
#include "ui_fittingwidget.h"
#include "pressurederivativecalculator.h"
#include <QtConcurrent>
#include <QMessageBox>
#include <QDebug>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <QFileDialog>
#include <QFile>
#include <QTextStream>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QHeaderView>

// ==========================================
// 引入 Eigen 库
// ==========================================
#include <Eigen/Dense>

// ===========================================================================
// FittingDataLoadDialog 实现
// ===========================================================================
FittingDataLoadDialog::FittingDataLoadDialog(const QList<QStringList>& previewData, QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle("数据列映射配置");
    resize(800, 550); //稍微调大高度以适应新选项
    setStyleSheet("QDialog { background-color: #f0f0f0; } QLabel, QComboBox, QPushButton, QTableWidget, QGroupBox { color: black; }");

    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->addWidget(new QLabel("请指定数据列含义 (时间必选):", this));

    m_previewTable = new QTableWidget(this);
    if(!previewData.isEmpty()) {
        int rows = qMin(previewData.size(), 50);
        int cols = previewData[0].size();
        m_previewTable->setRowCount(rows); m_previewTable->setColumnCount(cols);
        QStringList headers; for(int i=0;i<cols;++i) headers<<QString("Col %1").arg(i+1);
        m_previewTable->setHorizontalHeaderLabels(headers);
        for(int i=0;i<rows;++i) for(int j=0;j<cols && j<previewData[i].size();++j)
                m_previewTable->setItem(i,j,new QTableWidgetItem(previewData[i][j]));
    }
    m_previewTable->setAlternatingRowColors(true);
    layout->addWidget(m_previewTable);

    QGroupBox* grp = new QGroupBox("列映射与设置", this);
    QGridLayout* grid = new QGridLayout(grp);
    QStringList opts; for(int i=0;i<m_previewTable->columnCount();++i) opts<<QString("Col %1").arg(i+1);

    // 第一行：时间和压力列
    grid->addWidget(new QLabel("时间列 *:",this), 0, 0);
    m_comboTime = new QComboBox(this); m_comboTime->addItems(opts);
    grid->addWidget(m_comboTime, 0, 1);

    grid->addWidget(new QLabel("压力列:",this), 0, 2);
    m_comboPressure = new QComboBox(this); m_comboPressure->addItem("不导入",-1); m_comboPressure->addItems(opts);
    if(opts.size()>1) m_comboPressure->setCurrentIndex(2);
    grid->addWidget(m_comboPressure, 0, 3);

    // 第二行：导数和跳过行
    grid->addWidget(new QLabel("导数列:",this), 1, 0);
    m_comboDeriv = new QComboBox(this); m_comboDeriv->addItem("自动计算 (Bourdet)",-1); m_comboDeriv->addItems(opts);
    grid->addWidget(m_comboDeriv, 1, 1);

    grid->addWidget(new QLabel("跳过首行数:",this), 1, 2);
    m_comboSkipRows = new QComboBox(this);
    for(int i=0;i<=20;++i) m_comboSkipRows->addItem(QString::number(i),i);
    m_comboSkipRows->setCurrentIndex(1);
    grid->addWidget(m_comboSkipRows, 1, 3);

    // 第三行：数据类型选择（新增）
    grid->addWidget(new QLabel("压力数据类型:",this), 2, 0);
    m_comboPressureType = new QComboBox(this);
    m_comboPressureType->addItem("原始压力 (自动计算压差 |P-Pi|)", 0);
    m_comboPressureType->addItem("压差数据 (直接使用 ΔP)", 1);
    grid->addWidget(m_comboPressureType, 2, 1, 1, 3); // 跨3列

    layout->addWidget(grp);

    QHBoxLayout* btns = new QHBoxLayout;
    QPushButton* ok = new QPushButton("确定",this);
    QPushButton* cancel = new QPushButton("取消",this);
    connect(ok, &QPushButton::clicked, this, &FittingDataLoadDialog::validateSelection);
    connect(cancel, &QPushButton::clicked, this, &QDialog::reject);
    btns->addStretch(); btns->addWidget(ok); btns->addWidget(cancel);
    layout->addLayout(btns);
}

void FittingDataLoadDialog::validateSelection() { if(m_comboTime->currentIndex()<0) return; accept(); }
int FittingDataLoadDialog::getTimeColumnIndex() const { return m_comboTime->currentIndex(); }
int FittingDataLoadDialog::getPressureColumnIndex() const { return m_comboPressure->currentIndex()-1; }
int FittingDataLoadDialog::getDerivativeColumnIndex() const { return m_comboDeriv->currentIndex()-1; }
int FittingDataLoadDialog::getSkipRows() const { return m_comboSkipRows->currentData().toInt(); }
int FittingDataLoadDialog::getPressureDataType() const { return m_comboPressureType->currentData().toInt(); }


// ===========================================================================
// FittingWidget 实现
// ===========================================================================

FittingWidget::FittingWidget(QWidget *parent) : QWidget(parent), ui(new Ui::FittingWidget), m_modelManager(nullptr), m_isFitting(false)
{
    ui->setupUi(this);

    this->setStyleSheet("QWidget { color: black; font-family: Arial; } "
                        "QGroupBox { font-weight: bold; border: 1px solid gray; margin-top: 10px; } "
                        "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }");

    ui->splitter->setSizes(QList<int>{300, 800});
    ui->splitter->setCollapsible(0, false);
    ui->tableParams->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

    m_plot = new QCustomPlot(this);
    ui->plotContainer->layout()->addWidget(m_plot);
    setupPlot();

    qRegisterMetaType<QMap<QString,double>>("QMap<QString,double>");
    qRegisterMetaType<ModelManager::ModelType>("ModelManager::ModelType");
    qRegisterMetaType<QVector<double>>("QVector<double>");

    connect(this, &FittingWidget::sigIterationUpdated, this, &FittingWidget::onIterationUpdate, Qt::QueuedConnection);
    connect(this, &FittingWidget::sigProgress, ui->progressBar, &QProgressBar::setValue);
    connect(&m_watcher, &QFutureWatcher<void>::finished, this, &FittingWidget::onFitFinished);
}

FittingWidget::~FittingWidget() { delete ui; }
void FittingWidget::setModelManager(ModelManager *m) { m_modelManager = m; initModelCombo(); }
void FittingWidget::initModelCombo() {
    if(!m_modelManager) return;
    ui->comboModelSelect->clear();
    ui->comboModelSelect->addItems(ModelManager::getAvailableModelTypes());
    ui->comboModelSelect->setCurrentIndex((int)m_modelManager->getCurrentModelType());
    // 初始化时只加载参数，不绘图（刚打开界面为空）
    on_btnResetParams_clicked();
}

void FittingWidget::setupPlot() {
    m_plot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);
    m_plot->setBackground(Qt::white);

    // 设置 Log 坐标轴
    m_plot->xAxis->setScaleType(QCPAxis::stLogarithmic);
    m_plot->yAxis->setScaleType(QCPAxis::stLogarithmic);
    QSharedPointer<QCPAxisTickerLog> logTicker(new QCPAxisTickerLog);
    m_plot->xAxis->setTicker(logTicker);
    m_plot->yAxis->setTicker(logTicker);

    // 设置与 ModelWidget1 一致的格式和范围默认值
    m_plot->xAxis->setNumberFormat("eb"); m_plot->xAxis->setNumberPrecision(0);
    m_plot->xAxis->setLabel("t (h)");
    m_plot->yAxis->setLabel("Dp & dDp (MPa)");

    m_plot->xAxis->setRange(1e-3, 1e3);
    m_plot->yAxis->setRange(1e-3, 1e2);

    // Graph 0: 实测压力 (Red Circle)
    m_plot->addGraph();
    m_plot->graph(0)->setPen(Qt::NoPen);
    m_plot->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, Qt::red, 5));
    m_plot->graph(0)->setName("实测压力");

    // Graph 1: 实测导数 (Blue Triangle)
    m_plot->addGraph();
    m_plot->graph(1)->setPen(Qt::NoPen);
    m_plot->graph(1)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssTriangle, Qt::blue, 5));
    m_plot->graph(1)->setName("实测导数");

    // Graph 2: 理论压力 (Red Line)
    m_plot->addGraph();
    m_plot->graph(2)->setPen(QPen(Qt::red, 2));
    m_plot->graph(2)->setName("理论压力");

    // Graph 3: 理论导数 (Blue Line)
    m_plot->addGraph();
    m_plot->graph(3)->setPen(QPen(Qt::blue, 2));
    m_plot->graph(3)->setName("理论导数");

    m_plot->legend->setVisible(true);
}

void FittingWidget::setObservedData(const QVector<double>& t, const QVector<double>& p, const QVector<double>& d) {
    m_obsTime = t; m_obsPressure = p; m_obsDerivative = d;
    QVector<double> vt, vp, vd;
    for(int i=0; i<t.size(); ++i) {
        // Log坐标下只显示 > 0 的点
        if(t[i]>1e-6 && p[i]>1e-6) {
            vt<<t[i]; vp<<p[i];
            if(i<d.size() && d[i]>1e-6) vd<<d[i]; else vd<<1e-10;
        }
    }
    m_plot->graph(0)->setData(vt, vp);
    m_plot->graph(1)->setData(vt, vd);

    m_plot->rescaleAxes();

    if(m_plot->xAxis->range().lower<=0) m_plot->xAxis->setRangeLower(1e-3);
    if(m_plot->yAxis->range().lower<=0) m_plot->yAxis->setRangeLower(1e-3);

    m_plot->replot();
}

void FittingWidget::on_comboModelSelect_currentIndexChanged(int) {
    on_btnResetParams_clicked();
}

QString FittingWidget::getParamDisplayName(const QString& key) {
    if (key == "kf") return "内区渗透率 (kf)";
    if (key == "km") return "外区渗透率 (km)";
    if (key == "L") return "水平井长度 (L)";
    if (key == "Lf") return "裂缝半长 (Lf)";
    if (key == "LfD") return "无因次缝长 (LfD)";
    if (key == "rmD") return "复合半径 (rmD)";
    if (key == "omega1") return "内区储容比 (ω1)";
    if (key == "omega2") return "外区储容比 (ω2)";
    if (key == "lambda1") return "窜流系数 (λ1)";
    if (key == "cD") return "井筒储存 (Cd)";
    if (key == "S") return "表皮系数 (S)";
    if (key == "omega") return "储容比 (ω)";
    if (key == "lambda") return "窜流系数 (λ)";
    if (key == "k") return "渗透率 (k)";
    if (key == "Xf") return "裂缝半长 (Xf)";
    return key;
}

QStringList FittingWidget::getParamOrder(ModelManager::ModelType type) {
    QStringList order;
    if (type == ModelManager::InfiniteConductive) {
        order << "kf" << "km" << "L" << "Lf" << "LfD" << "rmD"
              << "omega1" << "omega2" << "lambda1" << "cD" << "S";
    } else if (type == ModelManager::FiniteConductive) {
        order << "omega" << "lambda" << "mf" << "nf" << "Xf"
              << "yy" << "y" << "CFD" << "kpd" << "cD" << "S";
    } else if (type == ModelManager::SegmentedMultiCluster) {
        order << "omega1" << "omega2" << "lambda1" << "lambda2"
              << "mf1" << "mf2" << "nf" << "Xf1" << "Xf2"
              << "yy1" << "yy2" << "y" << "CFD1" << "CFD2"
              << "kpd" << "cD" << "S";
    }
    return order;
}

void FittingWidget::on_btnResetParams_clicked() {
    if(!m_modelManager) return;
    ModelManager::ModelType type = (ModelManager::ModelType)ui->comboModelSelect->currentIndex();
    QMap<QString,double> defs = m_modelManager->getDefaultParameters(type);

    m_parameters.clear();
    QStringList orderedKeys = getParamOrder(type);

    for(const QString& key : orderedKeys) {
        if(defs.contains(key)) {
            FitParameter p;
            p.name = key;
            p.displayName = getParamDisplayName(p.name);
            p.value = defs[key];

            if(p.value > 0) { p.min = p.value * 0.001; p.max = p.value * 1000.0; }
            else if (p.value == 0) { p.min = 0.0; p.max = 100.0; }
            else { p.min = -100.0; p.max = 100.0; }

            p.isFit = false;
            m_parameters.append(p);
        }
    }
    loadParamsToTable();

    m_plot->graph(2)->data()->clear();
    m_plot->graph(3)->data()->clear();
    m_plot->replot();
}

void FittingWidget::loadParamsToTable() {
    ui->tableParams->setRowCount(0);
    ui->tableParams->blockSignals(true);
    for(int i=0; i<m_parameters.size(); ++i) {
        ui->tableParams->insertRow(i);
        QTableWidgetItem* name = new QTableWidgetItem(m_parameters[i].displayName);
        name->setData(Qt::UserRole, m_parameters[i].name); name->setFlags(name->flags()^Qt::ItemIsEditable);
        ui->tableParams->setItem(i,0,name);
        ui->tableParams->setItem(i,1,new QTableWidgetItem(QString::number(m_parameters[i].value)));
        QTableWidgetItem* chk = new QTableWidgetItem(); chk->setFlags(Qt::ItemIsUserCheckable|Qt::ItemIsEnabled|Qt::ItemIsSelectable);
        chk->setCheckState(m_parameters[i].isFit?Qt::Checked:Qt::Unchecked); ui->tableParams->setItem(i,2,chk);
        ui->tableParams->setItem(i,3,new QTableWidgetItem(QString("[%1,%2]").arg(m_parameters[i].min).arg(m_parameters[i].max)));
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
    // 获取数据类型：0=原始压力，1=压差数据
    int pressureType = dlg.getPressureDataType();

    QVector<double> t, p, d;
    double p_init = 0;

    // 如果选择是原始压力，则自动寻找初始压力（第一行数据）
    if(pressureType == 0 && pCol>=0) {
        for(int i=dlg.getSkipRows(); i<data.size(); ++i) {
            if(pCol<data[i].size()) {
                p_init = data[i][pCol].toDouble();
                break;
            }
        }
    }

    for(int i=dlg.getSkipRows(); i<data.size(); ++i) {
        if(tCol<data[i].size()) {
            double tv = data[i][tCol].toDouble();

            double pv = 0;
            if (pCol>=0 && pCol<data[i].size()) {
                double val = data[i][pCol].toDouble();
                if (pressureType == 0) {
                    // 原始压力模式：计算压差的绝对值
                    pv = std::abs(val - p_init);
                } else {
                    // 压差模式：直接使用
                    pv = val;
                }
            }

            if(tv>0) { t<<tv; p<<pv; }
        }
    }

    if (dCol >= 0) {
        for(int i=dlg.getSkipRows(); i<data.size(); ++i) {
            if(tCol<data[i].size() && data[i][tCol].toDouble() > 0 && dCol<data[i].size()) d << data[i][dCol].toDouble();
        }
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
    (void)QtConcurrent::run([this, modelType, paramsCopy](){ runOptimizationTask(modelType, paramsCopy); });
}

void FittingWidget::runOptimizationTask(ModelManager::ModelType modelType, QList<FitParameter> fitParams) {
    runLevenbergMarquardtOptimization(modelType, fitParams);
}

void FittingWidget::on_btnStop_clicked() { m_stopRequested=true; }

void FittingWidget::on_btnImportModel_clicked() {
    updateModelCurve();
}

void FittingWidget::updateModelCurve() {
    if(!m_modelManager) {
        QMessageBox::critical(this, "错误", "ModelManager 未初始化！请检查 MainWindow 中是否调用了 setModelManager。");
        return;
    }

    QWidget* focusWidget = QApplication::focusWidget();
    if (focusWidget && ui->tableParams->isAncestorOf(focusWidget)) {
        ui->tableParams->setFocus();
    }
    ui->tableParams->clearFocus();

    updateParamsFromTable();

    QMap<QString,double> currentParams;
    for(const auto& p : m_parameters) currentParams.insert(p.name, p.value);

    ModelManager::ModelType type = (ModelManager::ModelType)ui->comboModelSelect->currentIndex();

    QVector<double> targetT = m_obsTime;
    if(targetT.isEmpty()) {
        targetT.clear();
        for(double e = -4; e <= 4; e += 0.1) targetT.append(pow(10, e));
    }

    ModelCurveData res = m_modelManager->calculateTheoreticalCurve(type, currentParams, targetT);
    onIterationUpdate(0, currentParams, std::get<0>(res), std::get<1>(res), std::get<2>(res));
}

void FittingWidget::runLevenbergMarquardtOptimization(ModelManager::ModelType modelType, QList<FitParameter> params) {
    if(m_modelManager) m_modelManager->setHighPrecision(false);

    QVector<int> fitIndices;
    for(int i=0; i<params.size(); ++i) if(params[i].isFit) fitIndices.append(i);
    int nParams = fitIndices.size();
    if(nParams == 0) { QMetaObject::invokeMethod(this, "onFitFinished"); return; }

    double lambda = 0.01;
    int maxIter = 50;
    double currentSSE = 1e15;

    QMap<QString, double> currentParamMap;
    for(const auto& p : params) currentParamMap.insert(p.name, p.value);

    QVector<double> residuals = calculateResiduals(currentParamMap, modelType);
    currentSSE = calculateSumSquaredError(residuals);

    ModelCurveData curve = m_modelManager->calculateTheoreticalCurve(modelType, currentParamMap);
    emit sigIterationUpdated(currentSSE/residuals.size(), currentParamMap, std::get<0>(curve), std::get<1>(curve), std::get<2>(curve));

    for(int iter = 0; iter < maxIter; ++iter) {
        if(m_stopRequested) break;
        emit sigProgress(iter * 100 / maxIter);

        QVector<QVector<double>> J = computeJacobian(currentParamMap, residuals, fitIndices, modelType, params);
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
                double oldVal = params[pIdx].value;
                bool isLog = (oldVal > 1e-12 && pName != "S");

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

            QVector<double> newRes = calculateResiduals(trialMap, modelType);
            double newSSE = calculateSumSquaredError(newRes);

            if(newSSE < currentSSE) {
                currentSSE = newSSE;
                currentParamMap = trialMap;
                residuals = newRes;
                for(int i=0; i<nParams; ++i) params[fitIndices[i]].value = currentParamMap[params[fitIndices[i]].name];

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
    ModelCurveData finalCurve = m_modelManager->calculateTheoreticalCurve(modelType, currentParamMap);
    emit sigIterationUpdated(currentSSE/residuals.size(), currentParamMap, std::get<0>(finalCurve), std::get<1>(finalCurve), std::get<2>(finalCurve));

    QMetaObject::invokeMethod(this, "onFitFinished");
}

QVector<double> FittingWidget::calculateResiduals(const QMap<QString, double>& params, ModelManager::ModelType modelType) {
    if(!m_modelManager || m_obsTime.isEmpty()) return QVector<double>();

    QVector<double> targetT = m_obsTime;
    ModelCurveData res = m_modelManager->calculateTheoreticalCurve(modelType, params, targetT);
    const QVector<double>& pCal = std::get<1>(res);
    const QVector<double>& dpCal = std::get<2>(res);

    QVector<double> r;
    int count = qMin(m_obsPressure.size(), pCal.size());
    for(int i=0; i<count; ++i) {
        if(m_obsPressure[i] > 1e-10 && pCal[i] > 1e-10)
            r.append(log(m_obsPressure[i]) - log(pCal[i]));
        else
            r.append(0.0);
    }
    int dCount = qMin(m_obsDerivative.size(), dpCal.size());
    dCount = qMin(dCount, count);
    for(int i=0; i<dCount; ++i) {
        if(m_obsDerivative[i] > 1e-10 && dpCal[i] > 1e-10)
            r.append(log(m_obsDerivative[i]) - log(dpCal[i]));
        else
            r.append(0.0);
    }
    return r;
}

QVector<QVector<double>> FittingWidget::computeJacobian(const QMap<QString, double>& params, const QVector<double>& baseResiduals, const QVector<int>& fitIndices, ModelManager::ModelType modelType, const QList<FitParameter>& currentFitParams) {
    int nRes = baseResiduals.size();
    int nParams = fitIndices.size();
    QVector<QVector<double>> J(nRes, QVector<double>(nParams));

    for(int j = 0; j < nParams; ++j) {
        int idx = fitIndices[j];
        QString pName = currentFitParams[idx].name;
        double val = params.value(pName);
        bool isLog = (val > 1e-12 && pName != "S");
        double h;
        QMap<QString, double> pPlus = params;
        QMap<QString, double> pMinus = params;

        if(isLog) {
            h = 0.01;
            double valLog = log10(val);
            pPlus[pName] = pow(10.0, valLog + h);
            pMinus[pName] = pow(10.0, valLog - h);
        } else {
            h = 1e-4;
            pPlus[pName] = val + h;
            pMinus[pName] = val - h;
        }

        QVector<double> rPlus = calculateResiduals(pPlus, modelType);
        QVector<double> rMinus = calculateResiduals(pMinus, modelType);

        if(rPlus.size() == nRes && rMinus.size() == nRes) {
            for(int i=0; i<nRes; ++i) {
                J[i][j] = (rPlus[i] - rMinus[i]) / (2.0 * h);
            }
        }
    }
    return J;
}

QVector<double> FittingWidget::solveLinearSystem(const QVector<QVector<double>>& A, const QVector<double>& b) {
    int n = b.size();
    if (n == 0) return QVector<double>();
    Eigen::MatrixXd matA(n, n);
    Eigen::VectorXd vecB(n);
    for (int i = 0; i < n; ++i) {
        vecB(i) = b[i];
        for (int j = 0; j < n; ++j) matA(i, j) = A[i][j];
    }
    Eigen::VectorXd x = matA.ldlt().solve(vecB);
    QVector<double> res(n);
    for (int i = 0; i < n; ++i) res[i] = x(i);
    return res;
}

double FittingWidget::calculateSumSquaredError(const QVector<double>& residuals) {
    double sse = 0.0;
    for(double v : residuals) sse += v*v;
    return sse;
}

double FittingWidget::calculateError(const QMap<QString,double>& trialParams, ModelManager::ModelType modelType) {
    QVector<double> r = calculateResiduals(trialParams, modelType);
    if(r.isEmpty()) return 0;
    return calculateSumSquaredError(r) / r.size();
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
            if(i < m_parameters.size() && m_parameters[i].name == key) m_parameters[i].value = val;
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
        m_plot->graph(2)->setData(vt, vp);
        m_plot->graph(3)->setData(vt, vd);

        if (m_obsTime.isEmpty() && !vt.isEmpty()) {
            m_plot->rescaleAxes();
            if(m_plot->xAxis->range().lower<=0) m_plot->xAxis->setRangeLower(1e-3);
            if(m_plot->yAxis->range().lower<=0) m_plot->yAxis->setRangeLower(1e-3);
        }

        m_plot->replot();
    }
}
