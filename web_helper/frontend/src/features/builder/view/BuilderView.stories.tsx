import type { Meta, StoryObj } from '@storybook/react';
import { BuilderView } from './BuilderView';

const meta: Meta<typeof BuilderView> = {
  title: 'Features/Builder/View',
  component: BuilderView,
};

export default meta;

type Story = StoryObj<typeof BuilderView>;

export const Default: Story = {};
