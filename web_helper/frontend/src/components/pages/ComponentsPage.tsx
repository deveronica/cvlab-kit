import React from "react";

/**
 * Component Discovery Page - Browse and explore CVLab-Kit components
 */

import { useState } from 'react';
import { Search, Package, Layers, Code, BookOpen} from 'lucide-react';
import { useComponents, useComponentSearch, useComponentStats } from '../../hooks/useComponents';
import { Badge } from '../ui/badge';
import { AdvancedDataTable } from '../ui/advanced-data-table';
import type { ComponentInfo, ComponentCategory } from '../../lib/types';
import type { ColumnDef } from '@tanstack/react-table';

export function ComponentsPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedComponent, setSelectedComponent] = useState<ComponentInfo | null>(null);

  const { data: categories = [], isLoading: categoriesLoading } = useComponents();
  const { data: searchResults = [], isLoading: searchLoading } = useComponentSearch(searchQuery);
  const { totalComponents, categoryCounts } = useComponentStats();

  // Get all components for display
  const allComponents: ComponentInfo[] = React.useMemo(() => {
    if (searchQuery) {
      return searchResults;
    }

    const components: ComponentInfo[] = [];
    categories.forEach((category: ComponentCategory) => {
      if (selectedCategory === 'all' || selectedCategory === category.category) {
        components.push(...category.components);
      }
    });
    return components;
  }, [categories, searchResults, searchQuery, selectedCategory]);

  const columns: ColumnDef<ComponentInfo>[] = [
    {
      accessorKey: 'name',
      header: 'Name',
      cell: ({ getValue, row }) => (
        <div className="flex items-center space-x-2">
          <div className="flex h-8 w-8 items-center justify-center rounded bg-blue-100">
            <Package className="h-4 w-4 text-blue-600" />
          </div>
          <div>
            <div className="font-medium">{getValue<string>()}</div>
            <div className="text-xs text-muted-foreground">{row.original.type}</div>
          </div>
        </div>
      ),
    },
    {
      accessorKey: 'type',
      header: 'Category',
      cell: ({ getValue }) => (
        <Badge variant="outline" className="capitalize">
          {getValue<string>()}
        </Badge>
      ),
    },
    {
      accessorKey: 'description',
      header: 'Description',
      cell: ({ getValue }) => (
        <div className="max-w-md truncate text-sm text-muted-foreground">
          {getValue<string>() || 'No description available'}
        </div>
      ),
    },
    {
      id: 'parameters',
      header: 'Parameters',
      cell: ({ row }) => (
        <span className="text-sm text-muted-foreground">
          {Object.keys(row.original.parameters).length} params
        </span>
      ),
    },
    {
      id: 'examples',
      header: 'Examples',
      cell: ({ row }) => (
        <span className="text-sm text-muted-foreground">
          {row.original.examples.length} examples
        </span>
      ),
    },
    {
      id: 'actions',
      header: 'Actions',
      cell: ({ row }) => (
        <button
          onClick={() => setSelectedComponent(row.original)}
          className="text-sm text-blue-600 hover:text-blue-800 hover:underline"
        >
          View Details
        </button>
      ),
    },
  ];

  const handleCategoryChange = (category: string) => {
    setSelectedCategory(category);
    setSearchQuery(''); // Clear search when changing category
  };

  if (categoriesLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Component Discovery</h1>
          <p className="text-muted-foreground mt-1">
            Explore and browse available CVLab-Kit components
          </p>
        </div>
        <div className="flex items-center space-x-4">
          <div className="text-sm text-muted-foreground">
            {totalComponents} components available
          </div>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        {Object.entries(categoryCounts).map(([category, count]) => (
          <div
            key={category}
            className={`p-4 rounded-lg border cursor-pointer transition-colors ${
              selectedCategory === category
                ? 'border-blue-500 bg-blue-50'
                : 'hover:border-gray-300'
            }`}
            onClick={() => handleCategoryChange(category)}
          >
            <div className="flex items-center space-x-2">
              <Layers className="h-4 w-4 text-blue-600" />
              <span className="text-sm font-medium capitalize">{category}</span>
            </div>
            <div className="text-2xl font-bold mt-1">{count}</div>
          </div>
        ))}
        <div
          className={`p-4 rounded-lg border cursor-pointer transition-colors ${
            selectedCategory === 'all'
              ? 'border-blue-500 bg-blue-50'
              : 'hover:border-gray-300'
          }`}
          onClick={() => handleCategoryChange('all')}
        >
          <div className="flex items-center space-x-2">
            <Package className="h-4 w-4 text-gray-600" />
            <span className="text-sm font-medium">All</span>
          </div>
          <div className="text-2xl font-bold mt-1">{totalComponents}</div>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="flex items-center space-x-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search components..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 pr-4 py-2 w-full border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        {searchQuery && (
          <div className="text-sm text-muted-foreground">
            {searchLoading ? 'Searching...' : `${allComponents.length} results`}
          </div>
        )}
      </div>

      {/* Components Table */}
      <div className="bg-white rounded-lg shadow-sm border">
        <AdvancedDataTable columns={columns} data={allComponents} />
      </div>

      {/* Component Details Modal */}
      {selectedComponent && (
        <ComponentDetailsModal
          component={selectedComponent}
          onClose={() => setSelectedComponent(null)}
        />
      )}
    </div>
  );
}

// Component Details Modal
interface ComponentDetailsModalProps {
  component: ComponentInfo;
  onClose: () => void;
}

function ComponentDetailsModal({ component, onClose }: ComponentDetailsModalProps) {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <div className="flex items-center space-x-3">
            <div className="flex h-10 w-10 items-center justify-center rounded bg-blue-100">
              <Package className="h-5 w-5 text-blue-600" />
            </div>
            <div>
              <h2 className="text-xl font-bold">{component.name}</h2>
              <Badge variant="outline" className="capitalize">
                {component.type}
              </Badge>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            ✕
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Description */}
          {component.description && (
            <div>
              <h3 className="text-lg font-semibold mb-2 flex items-center">
                <BookOpen className="h-4 w-4 mr-2" />
                Description
              </h3>
              <p className="text-gray-600">{component.description}</p>
            </div>
          )}

          {/* Parameters */}
          <div>
            <h3 className="text-lg font-semibold mb-2 flex items-center">
              <Code className="h-4 w-4 mr-2" />
              Parameters ({Object.keys(component.parameters).length})
            </h3>
            {Object.keys(component.parameters).length > 0 ? (
              <div className="space-y-2">
                {Object.entries(component.parameters).map(([name, info]: [string, any]) => (
                  <div key={name} className="flex items-center justify-between p-3 bg-gray-50 rounded">
                    <div>
                      <code className="text-sm font-mono text-blue-600">{name}</code>
                      <span className="text-xs text-gray-500 ml-2">
                        {info.type} {info.required && '(required)'}
                      </span>
                    </div>
                    {info.default && (
                      <code className="text-xs bg-gray-200 px-2 py-1 rounded">
                        default: {info.default}
                      </code>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500">No parameters defined</p>
            )}
          </div>

          {/* Examples */}
          <div>
            <h3 className="text-lg font-semibold mb-2 flex items-center">
              <Code className="h-4 w-4 mr-2" />
              Configuration Examples ({component.examples.length})
            </h3>
            {component.examples.length > 0 ? (
              <div className="space-y-4">
                {component.examples.map((example, index) => (
                  <div key={index} className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
                    <pre className="text-sm">
                      <code>{JSON.stringify(example, null, 2)}</code>
                    </pre>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500">No examples available</p>
            )}
          </div>

          {/* File Path */}
          <div>
            <h3 className="text-lg font-semibold mb-2">File Location</h3>
            <code className="text-sm bg-gray-100 px-3 py-2 rounded block">
              {component.path}
            </code>
          </div>
        </div>
      </div>
    </div>
  );
}