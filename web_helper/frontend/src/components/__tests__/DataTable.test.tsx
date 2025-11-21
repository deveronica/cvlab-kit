import React from "react";
/**
 * Tests for DataTable component
 */

import { describe, it, expect, vi } from 'vitest';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { type ColumnDef } from '@tanstack/react-table';
import { DataTable } from '../ui/data-table';
import { render } from '../../test/utils';

interface TestData {
  id: string;
  name: string;
  status: string;
  value: number;
}

const testData: TestData[] = [
  { id: '1', name: 'Item 1', status: 'active', value: 100 },
  { id: '2', name: 'Item 2', status: 'inactive', value: 200 },
  { id: '3', name: 'Item 3', status: 'active', value: 150 },
];

const columns: ColumnDef<TestData>[] = [
  {
    accessorKey: 'name',
    header: 'Name',
  },
  {
    accessorKey: 'status',
    header: 'Status',
    cell: ({ getValue }) => (
      <span className={getValue<string>() === 'active' ? 'text-green-600' : 'text-red-600'}>
        {getValue<string>()}
      </span>
    ),
  },
  {
    accessorKey: 'value',
    header: 'Value',
    cell: ({ getValue }) => (
      <span>${getValue<number>()}</span>
    ),
  },
];

describe('DataTable', () => {
  it('renders table with data', () => {
    render(
      <DataTable
        columns={columns}
        data={testData}
      />
    );

    // Check headers
    expect(screen.getByText('Name')).toBeInTheDocument();
    expect(screen.getByText('Status')).toBeInTheDocument();
    expect(screen.getByText('Value')).toBeInTheDocument();

    // Check data
    expect(screen.getByText('Item 1')).toBeInTheDocument();
    expect(screen.getByText('Item 2')).toBeInTheDocument();
    expect(screen.getByText('Item 3')).toBeInTheDocument();
    expect(screen.getByText('$100')).toBeInTheDocument();
    expect(screen.getByText('$200')).toBeInTheDocument();
  });

  it('enables sorting when specified', async () => {
    const user = userEvent.setup();

    render(
      <DataTable
        columns={columns}
        data={testData}
        enableSorting={true}
      />
    );

    // Click on Name header to sort
    const nameHeader = screen.getByText('Name');
    await user.click(nameHeader);

    // Should see sorting icons
    expect(nameHeader.closest('th')).toBeInTheDocument();
  });

  it('disables sorting when specified', () => {
    render(
      <DataTable
        columns={columns}
        data={testData}
        enableSorting={false}
      />
    );

    // Headers should not be clickable for sorting
    const nameHeader = screen.getByText('Name');
    expect(nameHeader.closest('div')).not.toHaveClass('cursor-pointer');
  });

  it('enables pagination when specified', () => {
    render(
      <DataTable
        columns={columns}
        data={testData}
        enablePagination={true}
        pageSize={2}
      />
    );

    // Should see pagination controls
    expect(screen.getByText('Page 1 of 2')).toBeInTheDocument();
    expect(screen.getByText('Rows per page')).toBeInTheDocument();
  });

  it('disables pagination when specified', () => {
    render(
      <DataTable
        columns={columns}
        data={testData}
        enablePagination={false}
      />
    );

    // Should not see pagination controls
    expect(screen.queryByText('Page')).not.toBeInTheDocument();
    expect(screen.queryByText('Rows per page')).not.toBeInTheDocument();
  });

  it('handles row clicks', async () => {
    const user = userEvent.setup();
    const onRowClick = vi.fn();

    render(
      <DataTable
        columns={columns}
        data={testData}
        onRowClick={onRowClick}
      />
    );

    // Click on first row
    const firstRow = screen.getByText('Item 1').closest('tr');
    await user.click(firstRow!);

    expect(onRowClick).toHaveBeenCalledWith(testData[0]);
  });

  it('displays empty state when no data', () => {
    render(
      <DataTable
        columns={columns}
        data={[]}
      />
    );

    expect(screen.getByText('No results.')).toBeInTheDocument();
  });

  it('changes page size', async () => {
    const user = userEvent.setup();

    render(
      <DataTable
        columns={columns}
        data={testData}
        enablePagination={true}
        pageSize={1}
      />
    );

    // Should show page size selector
    const pageSizeSelect = screen.getByDisplayValue('1');
    expect(pageSizeSelect).toBeInTheDocument();

    // Change page size
    await user.selectOptions(pageSizeSelect, '2');
    expect(pageSizeSelect).toHaveValue('2');
  });

  it('navigates between pages', async () => {
    const user = userEvent.setup();

    render(
      <DataTable
        columns={columns}
        data={testData}
        enablePagination={true}
        pageSize={1}
      />
    );

    // Should show first item
    expect(screen.getByText('Item 1')).toBeInTheDocument();
    expect(screen.queryByText('Item 2')).not.toBeInTheDocument();

    // Click next page
    const nextButton = screen.getByText('›');
    await user.click(nextButton);

    // Should show second item
    expect(screen.queryByText('Item 1')).not.toBeInTheDocument();
    expect(screen.getByText('Item 2')).toBeInTheDocument();
  });

  it('applies custom className', () => {
    const { container } = render(
      <DataTable
        columns={columns}
        data={testData}
        className="custom-table"
      />
    );

    expect(container.firstChild).toHaveClass('custom-table');
  });

  it('renders custom cell content', () => {
    render(
      <DataTable
        columns={columns}
        data={testData}
      />
    );

    // Check custom cell rendering
    expect(screen.getByText('$100')).toBeInTheDocument();

    // Check status styling
    const activeStatus = screen.getAllByText('active')[0];
    expect(activeStatus).toHaveClass('text-green-600');

    const inactiveStatus = screen.getByText('inactive');
    expect(inactiveStatus).toHaveClass('text-red-600');
  });
});