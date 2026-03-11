import type { Meta, StoryObj } from '@storybook/react';
import { ThemeProvider, ExecuteProvider } from '@/app/ui';
import { ExecuteView } from './ExecuteView';

const meta: Meta<typeof ExecuteView> = {
  title: 'Features/Execute/View',
  component: ExecuteView,
  decorators: [
    (Story) => (
      <ThemeProvider>
        <ExecuteProvider>
          <Story />
        </ExecuteProvider>
      </ThemeProvider>
    ),
  ],
};

export default meta;

type Story = StoryObj<typeof ExecuteView>;

export const Default: Story = {};
