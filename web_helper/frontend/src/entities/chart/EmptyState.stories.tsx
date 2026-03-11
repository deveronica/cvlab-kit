import React from "react";
import type { Meta, StoryObj } from '@storybook/react';
import { EmptyState, InlineEmptyState } from './EmptyState';

const meta: Meta<typeof EmptyState> = {
  title: 'Charts/EmptyState',
  component: EmptyState,
  args: {
    variant: 'no-selection',
  },
};

export default meta;

type Story = StoryObj<typeof EmptyState>;

export const NoSelection: Story = {
  args: {
    variant: 'no-selection',
  },
};

export const NoData: Story = {
  args: {
    variant: 'no-data',
  },
};

export const NoMetric: Story = {
  args: {
    variant: 'no-metric',
    metricKey: 'val_accuracy',
  },
};

export const LoadingError: Story = {
  args: {
    variant: 'loading-error',
    actionLabel: 'Retry',
  },
};

export const Inline: StoryObj<typeof InlineEmptyState> = {
  render: () => <InlineEmptyState message="No data points to render." />,
};
