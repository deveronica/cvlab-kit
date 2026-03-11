import type { Meta, StoryObj } from '@storybook/react';
import { type ColumnDef } from '@tanstack/react-table';
import { AdvancedDataTable } from './advanced-data-table';

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

const pagedData = Array.from({ length: 12 }, (_, index) => ({
  id: `${index + 1}`,
  name: `Item ${index + 1}`,
  status: index % 2 === 0 ? 'active' : 'inactive',
  value: 100 + index,
}));

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
    cell: ({ getValue }) => <span>${getValue<number>()}</span>,
  },
];

const meta: Meta<typeof AdvancedDataTable> = {
  title: 'Components/AdvancedDataTable',
  component: AdvancedDataTable,
  args: {
    columns: columns as ColumnDef<unknown, unknown>[],
    data: testData as unknown[],
  },
};

export default meta;

type Story = StoryObj<typeof AdvancedDataTable>;

export const Default: Story = {};

export const SortingEnabled: Story = {
  args: {
    enableSorting: true,
  },
};

export const SortingDisabled: Story = {
  args: {
    enableSorting: false,
  },
};

export const PaginationEnabled: Story = {
  args: {
    enablePagination: true,
    pageSize: 10,
  },
};

export const PaginationDisabled: Story = {
  args: {
    enablePagination: false,
  },
};

export const RowClick: Story = {
  args: {
    onRowClick: () => {},
  },
};

export const EmptyState: Story = {
  args: {
    data: [],
  },
};

export const PageSizeChange: Story = {
  args: {
    enablePagination: true,
    pageSize: 10,
  },
};

export const PaginationNavigation: Story = {
  args: {
    data: pagedData as unknown[],
    enablePagination: true,
    pageSize: 10,
  },
};

export const CustomClassName: Story = {
  args: {
    className: 'custom-table',
  },
};

export const CustomCellContent: Story = {
  args: {
    columns: columns as ColumnDef<unknown, unknown>[],
    data: testData as unknown[],
  },
};
