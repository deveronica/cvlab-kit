import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react';
import { TabSwitcher } from './TabSwitcher';
import type { TabMode } from '@/entities/node-system/model/types';

const meta: Meta<typeof TabSwitcher> = {
  title: 'Node System/TabSwitcher',
  component: TabSwitcher,
  render: () => {
    const [tab, setTab] = useState<TabMode>('builder');
    return <TabSwitcher currentTab={tab} onTabChange={setTab} />;
  },
};

export default meta;

type Story = StoryObj<typeof TabSwitcher>;

export const Interactive: Story = {};
