import React from "react";
import type { Meta, StoryObj } from '@storybook/react';
import { ChartContainer, ChartLoadingSkeleton } from './ChartContainer';

const meta: Meta<typeof ChartContainer> = {
  title: 'Charts/ChartContainer',
  component: ChartContainer,
  args: {
    title: 'Sample Chart Container',
    description: 'Shared wrapper for chart cards',
    enableExport: false,
    enableFullscreen: false,
    children: (
      <div className="h-40 flex items-center justify-center border rounded-md text-sm text-muted-foreground">
        Chart content placeholder
      </div>
    ),
  },
};

export default meta;

type Story = StoryObj<typeof ChartContainer>;

export const Default: Story = {};

export const Loading: Story = {
  args: {
    isLoading: true,
  },
};

export const ErrorState: Story = {
  args: {
    error: new Error('Failed to load chart data'),
  },
};

export const LoadingSkeleton: StoryObj<typeof ChartLoadingSkeleton> = {
  render: () => <ChartLoadingSkeleton title="Loading Chart" description="Fetching chart data" />,
};
