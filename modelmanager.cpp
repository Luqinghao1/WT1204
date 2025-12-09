#include "modelmanager.h"
#include "modelwidget1.h"
#include "modelwidget2.h"
#include "modelwidget3.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QGroupBox>
#include <QDebug>
#include <cmath>

// ==================================================================================
//  ModelManager 构造与初始化
// ==================================================================================

ModelManager::ModelManager(QWidget* parent)
    : QObject(parent), m_mainWidget(nullptr), m_modelTypeCombo(nullptr), m_modelStack(nullptr)
    , m_modelWidget1(nullptr), m_modelWidget2(nullptr), m_modelWidget3(nullptr)
    , m_currentModelType(InfiniteConductive)
{
}

ModelManager::~ModelManager() {}

void ModelManager::initializeModels(QWidget* parentWidget)
{
    if (!parentWidget) return;
    createMainWidget();
    setupModelSelection();

    m_modelStack = new QStackedWidget(m_mainWidget);

    // 初始化子 Widget
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

    m_modelTypeCombo->addItem(getModelTypeName(InfiniteConductive));
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
    if (m_modelWidget1) connect(m_modelWidget1, &ModelWidget1::calculationCompleted, this, &ModelManager::onWidgetCalculationCompleted);
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
    case FiniteConductive: return "试井解释模型2";
    case SegmentedMultiCluster: return "试井解释模型3";
    default: return "未知模型";
    }
}

QStringList ModelManager::getAvailableModelTypes()
{
    return { getModelTypeName(InfiniteConductive), getModelTypeName(FiniteConductive), getModelTypeName(SegmentedMultiCluster) };
}

void ModelManager::onWidgetCalculationCompleted(const QString &t, const QMap<QString, double> &r) {
    emit calculationCompleted(t, r);
}

void ModelManager::setHighPrecision(bool high) {
    if (m_modelWidget1) m_modelWidget1->setHighPrecision(high);
}

QMap<QString, double> ModelManager::getDefaultParameters(ModelType type)
{
    QMap<QString, double> p;
    p.insert("cD", 0.001);
    p.insert("S", 0.01);
    p.insert("N", 4.0);

    if (type == InfiniteConductive) {
        p.insert("kf", 1e-3);
        p.insert("km", 1e-4);
        p.insert("L", 1000.0);
        p.insert("Lf", 100.0);
        p.insert("LfD", 0.1);
        p.insert("rmD", 4.0);
        p.insert("omega1", 0.4);
        p.insert("omega2", 0.08);
        p.insert("lambda1", 1e-3);
        p.insert("cD", 0.0);
        p.insert("S", 0.0);
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
    if (type == InfiniteConductive && m_modelWidget1) {
        return m_modelWidget1->calculateTheoreticalCurve(params, providedTime);
    }
    return ModelCurveData();
}

// === 新增：实测数据持久化接口实现 ===
void ModelManager::setObservedData(const QVector<double>& t, const QVector<double>& p, const QVector<double>& d)
{
    m_cachedObsTime = t;
    m_cachedObsPressure = p;
    m_cachedObsDerivative = d;
}

void ModelManager::getObservedData(QVector<double>& t, QVector<double>& p, QVector<double>& d) const
{
    t = m_cachedObsTime;
    p = m_cachedObsPressure;
    d = m_cachedObsDerivative;
}

bool ModelManager::hasObservedData() const
{
    return !m_cachedObsTime.isEmpty();
}
