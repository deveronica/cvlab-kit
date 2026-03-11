import type { Meta, StoryObj } from '@storybook/react';
import { Button } from './button';
import { fn } from '@storybook/test';

const meta = {
  title: 'Shared/UI/Button',
  component: Button,
  tags: ['autodocs'],
  argTypes: {
    variant: {
      control: 'select',
      options: ['default', 'destructive', 'outline', 'secondary', 'ghost', 'link'],
    },
    size: {
      control: 'select',
      options: ['default', 'sm', 'lg', 'icon'],
    },
    asChild: {
      table: {
        disable: true,
      },
    },
  },
  args: {
    onClick: fn(),
  },
} satisfies Meta<typeof Button>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    children: 'Button',
    variant: 'default',
  },
};

export const Secondary: Story = {
  args: {
    children: 'Secondary',
    variant: 'secondary',
  },
};

export const Destructive: Story = {
  args: {
    children: 'Destructive',
    variant: 'destructive',
  },
};

export const Outline: Story = {
  args: {
    children: 'Outline',
    variant: 'outline',
  },
};

export const Ghost: Story = {
  args: {
    children: 'Ghost',
    variant: 'ghost',
  },
};

export const Link: Story = {
  args: {
    children: 'Link',
    variant: 'link',
  },
};

export const Small: Story = {
  args: {
    children: 'Small',
    size: 'sm',
  },
};

export const Large: Story = {
  args: {
    children: 'Large',
    size: 'lg',
  },
};

export const Icon: Story = {
  args: {
    children: '🚀',
    size: 'icon',
  },
};
