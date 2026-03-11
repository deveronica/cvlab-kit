import React from "react";
import type { Meta, StoryObj } from '@storybook/react';
import { ChartExportButton } from './ChartExportButton';
import type { ChartExportOptions } from '@/shared/lib/charts/types';

const meta: Meta<typeof ChartExportButton> = {
  title: 'Charts/ChartExportButton',
  component: ChartExportButton,
  args: {
    onExport: async (_options: ChartExportOptions) => {
      await new Promise((resolve) => setTimeout(resolve, 300));
    },
  },
};

export default meta;

type Story = StoryObj<typeof ChartExportButton>;

export const Default: Story = {};

export const Disabled: Story = {
  args: {
    disabled: true,
  },
};

export const Icon: Story = {
  args: {
    size: 'sm',
  },
};
