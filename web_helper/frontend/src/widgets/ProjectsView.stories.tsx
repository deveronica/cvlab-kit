import type { Meta, StoryObj } from '@storybook/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { ProjectsView } from './ProjectsView';

const meta: Meta<typeof ProjectsView> = {
  title: 'Views/ProjectsView',
  component: ProjectsView,
  render: () => (
    <MemoryRouter initialEntries={['/projects/demo']}>
      <Routes>
        <Route path="/projects/:projectName" element={<ProjectsView />} />
      </Routes>
    </MemoryRouter>
  ),
};

export default meta;

type Story = StoryObj<typeof ProjectsView>;

export const Default: Story = {};
