import type { Meta, StoryObj } from '@storybook/react';
import { ThemeProvider } from '@/app/ui';
import { ExperimentsPage } from './ExperimentsPage';

const meta: Meta<typeof ExperimentsPage> = {
  title: 'Pages/ExperimentsPage',
  component: ExperimentsPage,
  decorators: [
    (Story) => (
      <ThemeProvider>
        <Story />
      </ThemeProvider>
    ),
  ],
};

export default meta;

type Story = StoryObj<typeof ExperimentsPage>;

export const Default: Story = {};
