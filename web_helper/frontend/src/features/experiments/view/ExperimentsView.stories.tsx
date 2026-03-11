import type { Meta, StoryObj } from '@storybook/react';
import { ThemeProvider } from '@/app/ui';
import { ExperimentsView } from './ExperimentsView';

const meta: Meta<typeof ExperimentsView> = {
  title: 'Features/Experiments/View',
  component: ExperimentsView,
  decorators: [
    (Story) => (
      <ThemeProvider>
        <Story />
      </ThemeProvider>
    ),
  ],
};

export default meta;

type Story = StoryObj<typeof ExperimentsView>;

export const Default: Story = {};
