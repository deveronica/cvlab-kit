import React from "react";
/**
 * Details Tab (FINAL FIX - Horizontal Scroll Guaranteed)
 *
 * Side-by-side comparison with WORKING horizontal scroll
 */

import { useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { InlineEmptyState } from '../../charts/EmptyState';
import type { Run } from '../../../lib/types';

interface ComparisonTableProps {
  title: string;
  description: string;
  runs: Run[];
  columns: string[];
  getData: (run: Run, column: string) => any;
}

function ComparisonTable({ title, description, runs, columns, getData }: ComparisonTableProps) {
  if (columns.length === 0) {
    return (
      <Card variant="compact">
        <CardContent variant="compact" className="py-8">
          <div className="text-center text-sm text-muted-foreground">
            No variations found. All values are identical.
          </div>
        </CardContent>
      </Card>
    );
  }

  // Calculate exact table width
  const tableWidth = 200 + columns.length * 180;

  return (
    <Card variant="compact">
      <CardHeader variant="compact">
        <CardTitle size="base">{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent variant="compact" className="p-0">
        {/* CRITICAL: This div MUST scroll horizontally */}
        <div
          className="border-t"
          style={{
            width: '100%',
            overflowX: 'scroll',
            overflowY: 'visible',
          }}
        >
          <table
            style={{
              width: `${tableWidth}px`,
              minWidth: `${tableWidth}px`,
              borderCollapse: 'collapse',
              fontSize: '0.875rem',
            }}
          >
            <thead>
              <tr style={{ borderBottom: '1px solid hsl(var(--border))' }}>
                {/* Sticky Run Name Column */}
                <th
                  style={{
                    position: 'sticky',
                    left: 0,
                    zIndex: 20,
                    // Fully opaque: solid card background + muted overlay
                    background: 'linear-gradient(hsl(var(--muted) / 0.5), hsl(var(--muted) / 0.5)), hsl(var(--card))',
                    textAlign: 'left',
                    padding: '12px',
                    fontWeight: 500,
                    width: '200px',
                    minWidth: '200px',
                    maxWidth: '200px',
                  }}
                >
                  <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    Run Name
                  </div>
                </th>
                {/* Data Columns */}
                {columns.map(col => (
                  <th
                    key={col}
                    style={{
                      // Fully opaque: solid card background + muted overlay
                      background: 'linear-gradient(hsl(var(--muted) / 0.5), hsl(var(--muted) / 0.5)), hsl(var(--card))',
                      textAlign: 'left',
                      padding: '12px',
                      fontWeight: 500,
                      width: '180px',
                      minWidth: '180px',
                      maxWidth: '180px',
                    }}
                  >
                    <div
                      style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                      title={col}
                    >
                      {col}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {runs.map((run, idx) => {
                // Use fully opaque backgrounds for sticky column
                const isEven = idx % 2 === 0;

                return (
                <tr
                  key={run.run_name}
                  style={{
                    borderBottom: '1px solid hsl(var(--border))',
                  }}
                  className="group transition-colors duration-200"
                >
                  {/* Sticky Run Name Cell */}
                  <td
                    className="group-hover:bg-muted/30"
                    style={{
                      position: 'sticky',
                      left: 0,
                      zIndex: 10,
                      // Layer: solid card background + stripe overlay
                      background: isEven
                        ? 'linear-gradient(hsl(var(--muted) / 0.1), hsl(var(--muted) / 0.1)), hsl(var(--card))'
                        : 'hsl(var(--card))',
                      padding: '12px',
                      fontFamily: 'monospace',
                      fontSize: '0.75rem',
                      width: '200px',
                      minWidth: '200px',
                      maxWidth: '200px',
                    }}
                  >
                    <div
                      style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                      title={run.run_name}
                    >
                      {run.run_name}
                    </div>
                  </td>
                  {/* Data Cells */}
                  {columns.map(col => {
                    const value = getData(run, col);
                    let display = '-';
                    let isNull = true;

                    if (value !== undefined && value !== null) {
                      isNull = false;
                      if (typeof value === 'number') {
                        display = value.toFixed(4);
                      } else if (typeof value === 'boolean') {
                        display = value ? 'true' : 'false';
                      } else if (typeof value === 'object') {
                        display = JSON.stringify(value);
                      } else {
                        display = String(value);
                      }
                    }

                    return (
                      <td
                        key={col}
                        className="group-hover:bg-muted/30"
                        style={{
                          background: isEven
                            ? 'linear-gradient(hsl(var(--muted) / 0.1), hsl(var(--muted) / 0.1)), hsl(var(--card))'
                            : 'hsl(var(--card))',
                          padding: '12px',
                          fontFamily: 'monospace',
                          fontSize: '0.75rem',
                          width: '180px',
                          minWidth: '180px',
                          maxWidth: '180px',
                          color: isNull ? 'hsl(var(--muted-foreground))' : 'inherit',
                        }}
                      >
                        <div
                          style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                          title={display}
                        >
                          {display}
                        </div>
                      </td>
                    );
                  })}
                </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

interface DetailsTabProps {
  runs: Run[];
  availableMetrics: string[];
}

export function DetailsTab({ runs, availableMetrics }: DetailsTabProps) {
  const allHyperparamKeys = useMemo(() => {
    const keys = new Set<string>();
    runs.forEach(run => {
      if (run.hyperparameters) {
        Object.keys(run.hyperparameters).forEach(key => keys.add(key));
      }
    });
    return Array.from(keys).sort();
  }, [runs]);

  const hasVariance = (values: any[]) => {
    if (values.length <= 1) return true;
    const validValues = values.filter(v => v !== undefined && v !== null);
    if (validValues.length === 0) return false;

    const first = validValues[0];
    if (typeof first === 'number') {
      return validValues.some(v => Math.abs((v as number) - (first as number)) > 1e-10);
    }
    return validValues.some(v => JSON.stringify(v) !== JSON.stringify(first));
  };

  const varyingHyperparams = useMemo(() => {
    return allHyperparamKeys.filter(key =>
      hasVariance(runs.map(r => r.hyperparameters?.[key]))
    );
  }, [allHyperparamKeys, runs]);

  const varyingMetrics = useMemo(() => {
    return availableMetrics.filter(key =>
      hasVariance(runs.map(r => r.metrics?.final?.[key]))
    ).sort();
  }, [availableMetrics, runs]);

  if (runs.length === 0) {
    return <InlineEmptyState message="No runs available for comparison" />;
  }

  return (
    <div className="space-y-4">
      {/* Info Banner */}
      <Card variant="compact">
        <CardContent variant="compact" className="py-3">
          <div className="flex items-center gap-2 text-sm flex-wrap">
            <Badge variant="outline">Diff Mode</Badge>
            <span className="text-muted-foreground">
              Showing {varyingHyperparams.length} hyperparams, {varyingMetrics.length} metrics with variations
            </span>
          </div>
        </CardContent>
      </Card>

      {/* Hyperparameters */}
      <ComparisonTable
        title="Hyperparameters"
        description={`${varyingHyperparams.length} varying hyperparameters across ${runs.length} runs`}
        runs={runs}
        columns={varyingHyperparams}
        getData={(run, col) => run.hyperparameters?.[col]}
      />

      {/* Metrics */}
      <ComparisonTable
        title="Final Metrics"
        description={`${varyingMetrics.length} varying metrics across ${runs.length} runs`}
        runs={runs}
        columns={varyingMetrics}
        getData={(run, col) => run.metrics?.final?.[col]}
      />
    </div>
  );
}
