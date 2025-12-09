#ifndef MODELWIDGET3_H
#define MODELWIDGET3_H

#include <QWidget>

namespace Ui {
class ModelWidget3;
}

class ModelWidget3 : public QWidget
{
    Q_OBJECT

public:
    explicit ModelWidget3(QWidget *parent = nullptr);
    ~ModelWidget3();

signals:
    void calculationCompleted(const QString &analysisType, const QMap<QString, double> &results);

private:
    Ui::ModelWidget3 *ui;
};

#endif // MODELWIDGET3_H
