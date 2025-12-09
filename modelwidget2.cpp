#include "modelwidget2.h"
#include "ui_modelwidget2.h"

ModelWidget2::ModelWidget2(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::ModelWidget2)
{
    ui->setupUi(this);
    // 此模型功能已移除，仅保留界面占位
}

ModelWidget2::~ModelWidget2()
{
    delete ui;
}
