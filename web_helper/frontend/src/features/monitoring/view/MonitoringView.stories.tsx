import type { Meta, StoryObj } from '@storybook/react';
import { MonitoringView } from './MonitoringView';

const meta: Meta<typeof MonitoringView> = {
  title: 'Features/Monitoring/View',
  component: MonitoringView,
};

export default meta;

type Story = StoryObj<typeof MonitoringView>;

export const Default: Story = {};
