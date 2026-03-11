import React from "react";
import type { Meta, StoryObj } from '@storybook/react';
import { useState } from 'react';
import { DataSeriesSelector } from './DataSeriesSelector';
import type { SeriesConfig } from '@/shared/lib/charts/types';

const initialSeries: SeriesConfig[] = [
  { dataKey: 'accuracy', name: 'Accuracy', color: '#3b82f6', visible: true, type: 'line' },
  { dataKey: 'loss', name: 'Loss', color: '#ef4444', visible: true, type: 'line' },
  { dataKey: 'f1', name: 'F1', color: '#10b981', visible: false, type: 'line' },
];

const meta: Meta<typeof DataSeriesSelector> = {
  title: 'Charts/DataSeriesSelector',
  component: DataSeriesSelector,
  render: (args) => {
    const [series, setSeries] = useState<SeriesConfig[]>(initialSeries);
    return (
      <div className="p-6">
        <DataSeriesSelector {...args} series={series} onSeriesChange={setSeries} />
      </div>
    );
  },
};

export default meta;

type Story = StoryObj<typeof DataSeriesSelector>;

export const Default: Story = {};

export const Compact: Story = {
  args: {
    compact: true,
  },
};
