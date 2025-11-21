import React from "react";

import { AlertCircle } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import type { OutlierDetectionResponse } from '@/lib/api/outliers';
import { X } from 'lucide-react';

interface OutlierDetectionModalProps {
  open: boolean;
  onClose: () => void;
  data: OutlierDetectionResponse;
  onOutlierSelect?: (runName: string) => void;
}

export function OutlierDetectionModal({
  open,
  onClose,
  data,
  onOutlierSelect,
}: OutlierDetectionModalProps) {
  const [activeTab, setActiveTab] = React.useState<'summary' | 'hyperparams' | 'metrics'>('summary');

  // Calculate outlier percentages
  const hyperparamOutlierPercentage = (data.hyperparameters.summary.total_outlier_runs / data.total_runs) * 100;
  const metricOutlierPercentage = (data.metrics.summary.total_outlier_runs / data.total_runs) * 100;

  // Convert outlier_counts to top_outlier_runs array
  const hyperparamTopOutliers = Object.entries(data.hyperparameters.summary.outlier_counts)
    .map(([run_name, outlier_count]) => ({ run_name, outlier_count }))
    .sort((a, b) => b.outlier_count - a.outlier_count);

  const _metricTopOutliers = Object.entries(data.metrics.summary.outlier_counts)
    .map(([run_name, outlier_count]) => ({ run_name, outlier_count }))
    .sort((a, b) => b.outlier_count - a.outlier_count);

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent hideCloseButton className="max-w-4xl h-[90vh] flex flex-col">
        <div className="flex flex-col space-y-1.5 text-left">
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-center gap-2">
              <AlertCircle className="h-5 w-5 text-orange-500 flex-shrink-0" />
              <DialogTitle>Outlier Detection - Detailed Analysis</DialogTitle>
            </div>
            <Button variant="ghost" size="sm" onClick={onClose} aria-label="Close">
              <X className="h-4 w-4" />
            </Button>
          </div>
          <DialogDescription>
            Statistical analysis of anomalous runs across hyperparameters and metrics
          </DialogDescription>
        </div>

        <div className="flex-1 overflow-hidden">
          <Tabs value={activeTab} onValueChange={(v: any) => setActiveTab(v)} className="h-full flex flex-col">
            <TabsList className="grid grid-cols-3 w-full">
              <TabsTrigger value="summary">Summary</TabsTrigger>
              <TabsTrigger value="hyperparams">Hyperparameters</TabsTrigger>
              <TabsTrigger value="metrics">Metrics</TabsTrigger>
            </TabsList>

            <div className="flex-1 overflow-y-auto mt-4">
              <TabsContent value="summary" className="space-y-3 mt-0">
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 border rounded-lg">
                    <div className="text-sm text-muted-foreground mb-1">
                      Hyperparameter Outliers
                    </div>
                    <div className="text-3xl font-bold">
                      {data.hyperparameters.summary.total_outlier_runs}
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">
                      {hyperparamOutlierPercentage.toFixed(1)}% of runs
                    </div>
                  </div>
                  <div className="p-4 border rounded-lg">
                    <div className="text-sm text-muted-foreground mb-1">
                      Metric Outliers
                    </div>
                    <div className="text-3xl font-bold">
                      {data.metrics.summary.total_outlier_runs}
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">
                      {metricOutlierPercentage.toFixed(1)}% of runs
                    </div>
                  </div>
                </div>

                <div className="space-y-2">
                  <h4 className="text-sm font-medium">Top Anomalous Runs (Hyperparameters)</h4>
                  <div className="space-y-2">
                    {hyperparamTopOutliers.slice(0, 10).map((item) => (
                      <div
                        key={item.run_name}
                        className="flex items-center justify-between p-3 border rounded-lg hover:bg-muted/50 cursor-pointer transition-colors"
                        onClick={() => {
                          onOutlierSelect?.(item.run_name);
                          onClose();
                        }}
                      >
                        <span className="text-sm font-mono truncate flex-1">
                          {item.run_name}
                        </span>
                        <Badge variant="destructive">
                          {item.outlier_count} columns
                        </Badge>
                      </div>
                    ))}
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="hyperparams" className="space-y-3 mt-0">
                {Object.entries(data.hyperparameters.column_results).map(([col, result]) => (
                  <div key={col} className="p-4 border rounded-lg space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium font-mono">{col}</span>
                      <Badge variant="outline">
                        {result.outlier_runs.length} outliers
                      </Badge>
                    </div>
                    {result.outlier_runs.length > 0 && (
                      <div className="space-y-2">
                        {result.outlier_runs.map((run) => (
                          <div
                            key={run}
                            className="text-sm font-mono p-2 border rounded hover:bg-muted/50 cursor-pointer"
                            onClick={() => {
                              onOutlierSelect?.(run);
                              onClose();
                            }}
                          >
                            → {run}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </TabsContent>

              <TabsContent value="metrics" className="space-y-3 mt-0">
                {Object.entries(data.metrics.column_results).map(([col, result]) => (
                  <div key={col} className="p-4 border rounded-lg space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium font-mono">{col}</span>
                      <Badge variant="outline">
                        {result.outlier_runs.length} outliers
                      </Badge>
                    </div>
                    {result.outlier_runs.length > 0 && (
                      <div className="space-y-2">
                        {result.outlier_runs.map((run) => (
                          <div
                            key={run}
                            className="text-sm font-mono p-2 border rounded hover:bg-muted/50 cursor-pointer"
                            onClick={() => {
                              onOutlierSelect?.(run);
                              onClose();
                            }}
                          >
                            → {run}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </TabsContent>
            </div>
          </Tabs>
        </div>
      </DialogContent>
    </Dialog>
  );
}
