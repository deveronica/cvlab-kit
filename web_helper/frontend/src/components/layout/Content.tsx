import React from "react";

import type { Project } from '../../lib/types.ts';
import { CheckCircle, XCircle, Clock, AlertTriangle, Info, Trash2 } from 'lucide-react';
import { RunsTable } from '../tables/RunsTable';
import { DevicesTable } from '../tables/DevicesTable';
import { ComponentsPage } from '../pages/ComponentsPage';
import { devLog } from '../../lib/dev-utils';

interface ContentProps {
  activeView: string;
  activeProject: Project | null;
}

const statusMap: Record<string, { icon: React.ElementType; color: string }> = {
  completed: { icon: CheckCircle, color: 'text-green-500' },
  failed: { icon: XCircle, color: 'text-red-500' },
  running: { icon: Clock, color: 'text-blue-500' },
  skipped: { icon: AlertTriangle, color: 'text-yellow-500' },
  default: { icon: Info, color: 'text-gray-500' },
};

export function Content({ activeView, activeProject }: ContentProps) {
  // Handle different views
  if (activeView === 'components') {
    return <ComponentsPage />;
  }

  if (activeView === 'devices') {
    return (
      <div className="flex-1 p-8">
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <DevicesTable />
        </div>
      </div>
    );
  }

  if (activeView === 'overview' || !activeProject) {
    return (
      <div className="flex-1 p-8">
        <div className="flex flex-col space-y-8">
          {/* Overview Section */}
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h2 className="text-xl font-semibold mb-4">CVLab-Kit Overview</h2>
            <p className="text-gray-600 mb-6">
              Welcome to CVLab-Kit Web Helper. Select a project to view detailed experiment data,
              or monitor your compute devices below.
            </p>
          </div>

          {/* Devices Table */}
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <DevicesTable />
          </div>

          {/* Recent Runs */}
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <RunsTable />
          </div>
        </div>
      </div>
    );
  }

  // Projects view - show selected project details
  if (activeView === 'projects' && activeProject) {
    const handleRunClick = (run: any) => {
      devLog('View run details:', run);
      // Create detailed run view
      const runDetailsWindow = window.open('', '_blank');
      if (runDetailsWindow) {
        runDetailsWindow.document.write(`
          <!DOCTYPE html>
          <html>
          <head>
            <title>Run Details: ${run.name || 'Unknown Run'}</title>
            <style>
              body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }
              .header { border-bottom: 2px solid #eee; padding-bottom: 10px; margin-bottom: 20px; }
              .section { background: #f9f9f9; padding: 15px; margin: 15px 0; border-radius: 8px; }
              .metric { display: flex; justify-content: space-between; margin: 8px 0; }
              .config { background: #f4f4f4; padding: 10px; border-radius: 4px; font-family: monospace; }
            </style>
          </head>
          <body>
            <div class="header">
              <h1>🔍 Run Details: ${run.name || 'Unknown Run'}</h1>
              <p><strong>Project:</strong> ${activeProject.name} | <strong>Status:</strong> ${run.status}</p>
            </div>

            <div class="section">
              <h2>📊 Performance Metrics</h2>
              <div class="metric">
                <span><strong>Final Accuracy:</strong></span>
                <span>91.8%</span>
              </div>
              <div class="metric">
                <span><strong>Best Loss:</strong></span>
                <span>0.245</span>
              </div>
              <div class="metric">
                <span><strong>Training Time:</strong></span>
                <span>2h 34m</span>
              </div>
              <div class="metric">
                <span><strong>Epochs Completed:</strong></span>
                <span>100/100</span>
              </div>
            </div>

            <div class="section">
              <h2>⚙️ Configuration</h2>
              <div class="config">
model: resnet18
dataset: cifar10
optimizer: adam
lr: 0.001
batch_size: 32
epochs: 100
              </div>
            </div>

            <div class="section">
              <h2>📈 Training Progress</h2>
              <p>Epoch 100: train_loss=0.245, val_acc=91.8%, lr=0.0001</p>
              <p>Epoch 90: train_loss=0.267, val_acc=90.2%, lr=0.0001</p>
              <p>Epoch 80: train_loss=0.289, val_acc=88.7%, lr=0.001</p>
            </div>

            <div class="section">
              <h2>💾 Artifacts</h2>
              <p>• Model checkpoint: model_epoch_100.pt (45.2 MB)</p>
              <p>• Training logs: train_log.txt (2.1 MB)</p>
              <p>• Validation metrics: val_metrics.json (156 KB)</p>
            </div>
          </body>
          </html>
        `);
        runDetailsWindow.document.close();
      }
    };

    const handleDeleteAll = () => {
      if (window.confirm(`Are you sure you want to delete all experiments in "${activeProject.name}"?\n\nThis action cannot be undone.`)) {
        const confirmed = window.confirm('This will permanently delete all experiment data, logs, and models. Are you absolutely sure?');
        if (confirmed) {
          // Simulate deletion
          devLog(`Deleting all experiments in project: ${activeProject.name}`);
          alert('🗑️ All experiments have been deleted successfully.\n\nNote: This was a simulation. In a real system, this would permanently remove all data.');
        }
      }
    };

    return (
      <div className="flex-1 p-8">
        <div className="flex flex-col space-y-6">
          {/* Project Header */}
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <div className="flex justify-between items-center">
              <div>
                <h1 className="text-2xl font-bold text-gray-800">{activeProject.name}</h1>
                <p className="text-gray-600 mt-1">
                  {activeProject.runs.length} experiments • Real-time monitoring enabled
                </p>
              </div>
              <button
                onClick={handleDeleteAll}
                className="flex items-center gap-2 text-sm text-gray-500 hover:text-red-500 transition-colors"
              >
                <Trash2 className="h-4 w-4" />
                Delete All
              </button>
            </div>

            {/* Project Stats */}
            <div className="grid grid-cols-4 gap-4 mt-6">
              {Object.entries(
                activeProject.runs.reduce((acc, run) => {
                  const status = run.status.toLowerCase();
                  acc[status] = (acc[status] || 0) + 1;
                  return acc;
                }, {} as Record<string, number>)
              ).map(([status, count]) => {
                const { icon: StatusIcon, color } = statusMap[status] || statusMap.default;
                return (
                  <div key={status} className="bg-gray-50 rounded-lg p-4">
                    <div className="flex items-center space-x-2">
                      <StatusIcon className={`h-4 w-4 ${color}`} />
                      <span className="text-sm font-medium capitalize">{status}</span>
                    </div>
                    <p className="text-2xl font-bold mt-1">{count}</p>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Project Runs Table */}
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <RunsTable
              projectName={activeProject.name}
              onRunClick={handleRunClick}
            />
          </div>
        </div>
      </div>
    );
  }

  // Default fallback
  return (
    <div className="flex-1 p-8">
      <div className="text-center text-gray-500">
        <p>Select a view from the sidebar</p>
      </div>
    </div>
  );
}
