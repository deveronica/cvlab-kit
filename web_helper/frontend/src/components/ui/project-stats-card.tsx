import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from './card';
import { TrendingUp, TrendingDown, Minus, LucideIcon } from 'lucide-react';
import { cn } from '../../lib/utils';

interface ProjectStatsCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: LucideIcon;
  trend?: {
    value: number;
    label?: string;
  };
  className?: string;
}

export function ProjectStatsCard({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  className,
}: ProjectStatsCardProps) {
  const getTrendIcon = () => {
    if (!trend) return null;
    if (trend.value > 0) return TrendingUp;
    if (trend.value < 0) return TrendingDown;
    return Minus;
  };

  const getTrendColor = () => {
    if (!trend) return '';
    if (trend.value > 0) return 'text-green-600 dark:text-green-400';
    if (trend.value < 0) return 'text-red-600 dark:text-red-400';
    return 'text-gray-500 dark:text-gray-400';
  };

  const TrendIcon = getTrendIcon();

  return (
    <Card className={cn('hover:shadow-md transition-shadow', className)}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {title}
        </CardTitle>
        {Icon && (
          <Icon className="h-4 w-4 text-muted-foreground" />
        )}
      </CardHeader>
      <CardContent>
        <div className="space-y-1">
          <div className="text-2xl font-bold">{value}</div>
          {(subtitle || trend) && (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              {trend && TrendIcon && (
                <div className={cn('flex items-center gap-1', getTrendColor())}>
                  <TrendIcon className="h-3 w-3" />
                  <span className="font-medium">
                    {Math.abs(trend.value)}%
                  </span>
                  {trend.label && <span className="text-muted-foreground">{trend.label}</span>}
                </div>
              )}
              {subtitle && !trend && <span>{subtitle}</span>}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

interface CompactStatsCardProps {
  stats: Array<{
    label: string;
    value: string | number;
    valueClassName?: string;
  }>;
  className?: string;
}

export function CompactStatsCard({ stats, className }: CompactStatsCardProps) {
  return (
    <Card className={cn('p-4', className)}>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {stats.map((stat, idx) => (
          <div key={idx} className="space-y-1">
            <div className="text-xs font-medium text-muted-foreground">{stat.label}</div>
            <div className={cn('text-xl font-bold', stat.valueClassName)}>{stat.value}</div>
          </div>
        ))}
      </div>
    </Card>
  );
}
