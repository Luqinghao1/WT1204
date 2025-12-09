#ifndef MODELWIDGET2_H
#define MODELWIDGET2_H

#include <QWidget>

namespace Ui {
class ModelWidget2;
}

class ModelWidget2 : public QWidget
{
    Q_OBJECT

public:
    explicit ModelWidget2(QWidget *parent = nullptr);
    ~ModelWidget2();

signals:
    void calculationCompleted(const QString &analysisType, const QMap<QString, double> &results);

private:
    Ui::ModelWidget2 *ui;
};

#endif // MODELWIDGET2_H
