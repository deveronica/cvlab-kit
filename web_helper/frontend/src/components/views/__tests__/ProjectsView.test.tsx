import React from "react";
import { render, screen } from '@testing-library/react';
import { ProjectsView } from '../ProjectsView';

describe('ProjectsView', () => {
  it('renders without crashing', () => {
    render(<ProjectsView activeProject={null} setActiveProject={() => {}} />);
    expect(screen.getByText('Projects')).toBeInTheDocument();
  });
});
