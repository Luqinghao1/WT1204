#ifndef MOUSEZOOM_H
#define MOUSEZOOM_H

#include "qcustomplot.h"

/**
 * @brief 增强型绘图控件 (MouseZoom)
 * 继承自 QCustomPlot，提供针对试井分析优化的交互体验。
 * 功能：
 * 1. 滚轮缩放：默认全向，按住左键纵向缩放，按住右键横向缩放。
 * 2. 提供便捷的图表配置接口。
 */
class MouseZoom : public QCustomPlot
{
    Q_OBJECT
public:
    explicit MouseZoom(QWidget *parent = nullptr);

protected:
    /**
     * @brief 重写滚轮事件
     * 实现定向缩放逻辑。
     */
    void wheelEvent(QWheelEvent *event) override;
};

#endif // MOUSEZOOM_H
