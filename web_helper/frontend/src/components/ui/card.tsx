import * as React from 'react'
import { cn } from '../../lib/utils'

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  draggable?: boolean;
  expandable?: boolean;
  onExpand?: () => void;
  variant?: 'default' | 'compact';
}

const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, draggable = false, expandable = false, onExpand, variant = 'default', ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        // Base card styles
        'bg-card border border-border rounded-lg overflow-hidden transition-colors duration-200',
        // Variant-specific styles
        variant === 'default' && [
          'shadow-sm p-1',
        ],
        variant === 'compact' && [
          'shadow-sm p-0',
        ],
        // Interactive states
        draggable && 'cursor-move hover:border-accent',
        expandable && 'cursor-pointer hover:border-accent hover:bg-accent/5',
        // Focus styles
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
        className
      )}
      onClick={expandable ? onExpand : undefined}
      tabIndex={expandable ? 0 : undefined}
      role={expandable ? "button" : undefined}
      {...props}
    />
  )
)
Card.displayName = 'Card'

interface CardHeaderProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'compact';
}

const CardHeader = React.forwardRef<HTMLDivElement, CardHeaderProps>(
  ({ className, variant = 'default', ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        'flex flex-col space-y-1.5',
        variant === 'default' ? 'p-6 pb-4' : 'p-4 pb-3',
        className
      )}
      {...props}
    />
  )
)
CardHeader.displayName = 'CardHeader'

interface CardTitleProps extends React.HTMLAttributes<HTMLHeadingElement> {
  size?: 'default' | 'lg' | 'base' | 'sm';
}

const CardTitle = React.forwardRef<HTMLParagraphElement, CardTitleProps>(
  ({ className, size = 'default', ...props }, ref) => (
    <h3
      ref={ref}
      className={cn(
        'font-semibold leading-none tracking-tight',
        size === 'default' && 'text-2xl',
        size === 'lg' && 'text-lg',
        size === 'base' && 'text-base',
        size === 'sm' && 'text-sm',
        className
      )}
      {...props}
    />
  )
)
CardTitle.displayName = 'CardTitle'

const CardDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <p
    ref={ref}
    className={cn('text-sm text-muted-foreground', className)}
    {...props}
  />
))
CardDescription.displayName = 'CardDescription'

interface CardContentProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'compact';
}

const CardContent = React.forwardRef<HTMLDivElement, CardContentProps>(
  ({ className, variant = 'default', ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        variant === 'default' ? 'p-6 pt-0' : 'p-3 pt-0',
        className
      )}
      {...props}
    />
  )
)
CardContent.displayName = 'CardContent'

const CardFooter = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn('flex items-center p-6 pt-0', className)}
    {...props}
  />
))
CardFooter.displayName = 'CardFooter'

export { Card, CardHeader, CardFooter, CardTitle, CardDescription, CardContent }