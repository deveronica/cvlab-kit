import React from "react";
import type { Meta, StoryObj } from '@storybook/react';
import { useState } from 'react';
import { ChartTypeSelector } from './ChartTypeSelector';
import type { ChartType } from '@/shared/lib/charts/types';

const meta: Meta<typeof ChartTypeSelector> = {
  title: 'Charts/ChartTypeSelector',
  component: ChartTypeSelector,
  render: (args) => {
    const [value, setValue] = useState<ChartType>('line');
    return (
      <div className="p-6">
        <ChartTypeSelector {...args} value={value} onChange={setValue} />
      </div>
    );
  },
};

export default meta;

type Story = StoryObj<typeof ChartTypeSelector>;

export const Default: Story = {};

export const LimitedTypes: Story = {
  args: {
    supportedTypes: ['line', 'bar', 'area'],
  },
};

export const RendererCompatibility: Story = {
  args: {
    currentRenderer: 'plotly',
  },
};
