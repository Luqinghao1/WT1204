#include "mousezoom.h"

MouseZoom::MouseZoom(QWidget *parent) : QCustomPlot(parent)
{
    // 初始化默认交互方式：支持拖拽和缩放
    setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables | QCP::iSelectLegend);

    // 设置默认背景色
    setBackground(Qt::white);
    axisRect()->setBackground(Qt::white);
}

void MouseZoom::wheelEvent(QWheelEvent *event)
{
    // 获取当前鼠标按键状态
    Qt::MouseButtons buttons = QApplication::mouseButtons();

    // 1. 先重置为全向缩放模式
    axisRect()->setRangeZoom(Qt::Horizontal | Qt::Vertical);

    // 2. 根据按键修改缩放模式
    // 如果按住左键：锁定X轴，仅缩放Y轴
    if (buttons & Qt::LeftButton)
        axisRect()->setRangeZoom(Qt::Vertical);
    // 如果按住右键：锁定Y轴，仅缩放X轴
    else if (buttons & Qt::RightButton)
        axisRect()->setRangeZoom(Qt::Horizontal);

    // 3. 调用父类处理实际缩放计算
    QCustomPlot::wheelEvent(event);

    // 4. 恢复默认设置，以免影响拖拽等其他交互
    axisRect()->setRangeZoom(Qt::Horizontal | Qt::Vertical);
}
