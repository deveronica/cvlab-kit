import type { Meta, StoryObj } from '@storybook/react';
import { MonitoringPage } from './MonitoringPage';

const meta: Meta<typeof MonitoringPage> = {
  title: 'Pages/MonitoringPage',
  component: MonitoringPage,
};

export default meta;

type Story = StoryObj<typeof MonitoringPage>;

export const Default: Story = {};
