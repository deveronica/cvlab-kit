import React from "react";
/**
 * Chart Theme Selector Component
 *
 * Dropdown for selecting chart color themes
 */

import { Palette } from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuLabel,
  DropdownMenuSeparator,
} from './dropdown-menu';
import { Button } from './button';
import { useChartTheme } from '../../contexts/ChartThemeContext';
import { getThemeMetadata } from '../../lib/chart-themes';

interface ChartThemeSelectorProps {
  size?: 'sm' | 'default';
  variant?: 'outline' | 'ghost' | 'default';
  className?: string;
}

export function ChartThemeSelector({
  size = 'sm',
  variant = 'outline',
  className = '',
}: ChartThemeSelectorProps) {
  const { themeName, setTheme, availableThemes } = useChartTheme();

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant={variant}
          size={size}
          className={className}
          title="Chart theme"
        >
          <Palette className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        <DropdownMenuLabel>Chart Theme</DropdownMenuLabel>
        <DropdownMenuSeparator />
        {Object.keys(availableThemes).map((themeKey) => {
          const metadata = getThemeMetadata(themeKey);
          const isActive = themeName === themeKey;

          return (
            <DropdownMenuItem
              key={themeKey}
              onClick={() => setTheme(themeKey)}
              className={isActive ? 'bg-accent' : ''}
            >
              <div className="flex items-center gap-3 w-full">
                <div className="flex gap-1">
                  {metadata.previewColors.map((color, i) => (
                    <div
                      key={i}
                      className="w-3 h-3 rounded-sm"
                      style={{ backgroundColor: color }}
                    />
                  ))}
                </div>
                <div className="flex-1">
                  <div className="font-medium text-sm">{metadata.displayName}</div>
                  <div className="text-xs text-muted-foreground">
                    {metadata.description}
                  </div>
                </div>
                {isActive && (
                  <div className="text-xs text-primary">✓</div>
                )}
              </div>
            </DropdownMenuItem>
          );
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
