import type { Meta, StoryObj } from '@storybook/react';
import { BuilderPage } from './BuilderPage';

const meta: Meta<typeof BuilderPage> = {
  title: 'Pages/BuilderPage',
  component: BuilderPage,
};

export default meta;

type Story = StoryObj<typeof BuilderPage>;

export const Default: Story = {};
