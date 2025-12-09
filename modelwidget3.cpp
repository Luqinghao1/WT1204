#include "modelwidget3.h"
#include "ui_modelwidget3.h"

ModelWidget3::ModelWidget3(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::ModelWidget3)
{
    ui->setupUi(this);
    // 此模型功能已移除，仅保留界面占位
}

ModelWidget3::~ModelWidget3()
{
    delete ui;
}
