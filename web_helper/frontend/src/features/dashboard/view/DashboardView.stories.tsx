import type { Meta, StoryObj } from '@storybook/react';
import { DashboardView } from './DashboardView';

const meta: Meta<typeof DashboardView> = {
  title: 'Features/Dashboard/View',
  component: DashboardView,
};

export default meta;

type Story = StoryObj<typeof DashboardView>;

export const Default: Story = {};
