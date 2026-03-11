import type { Meta, StoryObj } from '@storybook/react';
import { ThemeProvider, ExecuteProvider } from '@/app/ui';
import { ExecutePage } from './ExecutePage';

const meta: Meta<typeof ExecutePage> = {
  title: 'Pages/ExecutePage',
  component: ExecutePage,
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

type Story = StoryObj<typeof ExecutePage>;

export const Default: Story = {};
