import type { Meta, StoryObj } from '@storybook/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { ProjectsPage } from './ProjectsPage';

const meta: Meta<typeof ProjectsPage> = {
  title: 'Pages/ProjectsPage',
  component: ProjectsPage,
  render: () => (
    <MemoryRouter initialEntries={['/projects/demo']}>
      <Routes>
        <Route path="/projects/:projectName" element={<ProjectsPage />} />
      </Routes>
    </MemoryRouter>
  ),
};

export default meta;

type Story = StoryObj<typeof ProjectsPage>;

export const Default: Story = {};
