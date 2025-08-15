# TODO: The current implementation replaces the instance's class with a dynamically
#       generated one for delegation. This is clever but can be confusing and might
#       have side effects with type checking or serialization. A more explicit
#       approach using `__getattr__` on the instance itself could be considered.

from abc import ABCMeta, abstractmethod


class InterfaceMeta(ABCMeta):
    """A metaclass for creating flexible, PyTorch-compatible components.

    This metaclass enables two primary component implementation patterns by
    customizing the instantiation process (`__call__`):

    1.  **Direct Implementation:** A class inherits from a component ABC (like
        `Model` or `Optimizer`) and implements all its abstract methods. This is
        the standard object-oriented approach.

    2.  **Delegation (Composition):** A class inherits from a component ABC but
        does not implement all abstract methods. Instead, in its `__init__`,
        it creates an instance of an existing library class (e.g.,
        `torch.optim.Adam`) and assigns it to an instance attribute. This
        metaclass will automatically delegate any unimplemented method calls
        from the wrapper class to the contained library object.

    3.  **How it works:**
        - When a class using this metaclass is instantiated, `__call__` is invoked.
        - It first creates the instance, temporarily bypassing the standard ABC
          abstract method checks.
        - It then inspects the new instance's `__dict__` to find an attribute that
          is an instance of one of the PyTorch's base classes (e.g., finding a
          `torch.optim.Optimizer` instance if the class being instantiated
          inherits from `cvlabkit.component.base.Optimizer`).
        - If such a "delegation target" is found, it dynamically creates a new class
          on the fly. This new class inherits from that pre-existing delegation target
          and overrides the methods that were implemented in the original class.
        - Finally, it "swaps" the instance's original class to be this delegating class,
          allowing method calls to be automatically delegated to the target object.
          except for those methods that were explicitly implemented in the original class.
        - If no delegation target is found, it enforces the standard ABC rule,
          raising a `TypeError` if any abstract methods are not implemented.
    """

    def __call__(cls, cfg, *args, **kwargs):
        """Customizes the instantiation process to support delegation.

        This method is the core of the delegation logic. It intercepts the
        class instantiation call.

        Args:
            *args: Positional arguments for the class constructor.
            **kwargs: Keyword arguments for the class constructor.

        Returns:
            An instance of the class, potentially with a dynamically generated
            delegator class.
        """
        # Temporarily bypass the ABC check to allow instance creation first.
        # This is necessary so we can inspect the instance's attributes to find
        # a potential delegation target.
        original_abstract_methods = cls.__abstractmethods__
        cls.__abstractmethods__ = frozenset()
        instance = super().__call__(cfg, *args, **kwargs)
        # Restore the abstract methods to maintain class integrity for future use.
        cls.__abstractmethods__ = original_abstract_methods

        # --- Find a potential component to delegate to ---
        delegation_target = None
        # Look for an attribute on the instance that is an instance of a base class.
        # We check against the MRO (Method Resolution Order) to find valid bases,
        # excluding the class itself and the base `object`.
        base_classes = tuple(b for b in cls.__mro__[1:] if b is not object)
        if base_classes:
            for attr_value in instance.__dict__.values():
                if isinstance(attr_value, base_classes):
                    delegation_target = attr_value
                    break

        if delegation_target:
            # --- Case 2: Delegation Pattern ---
            # A component to delegate to was found. Now, we create a dynamic
            # wrapper class to handle the delegation automatically.

            # Store the target on the instance using a private name to avoid
            # conflicts. The dynamically generated methods will use this.
            setattr(instance, '_delegated_component', delegation_target)

            delegating_methods = {}
            # Find all methods that need to be delegated.
            for name in dir(delegation_target):
                if name.startswith('__'):
                    continue  # Skip magic methods

                # We only need to create a delegate if the method is not already
                # implemented on the wrapper class itself. This allows users to
                # override specific methods while delegating the rest.
                has_concrete_implementation = hasattr(cls, name) and not getattr(
                    getattr(cls, name), "__isabstractmethod__", False
                )

                if not has_concrete_implementation:
                    # This lambda captures the method name and creates a function
                    # that calls the corresponding method on the delegated component.
                    delegating_methods[name] = (
                        lambda self, *a, _name=name, **kw:
                            getattr(self._delegated_component, _name)(*a, **kw)
                    )

            # Create a new class type on the fly.
            # It inherits from the original class and adds the delegating methods.
            # The new class is named to reflect its dynamic and delegated nature.
            DynamicDelegator = type(
                f"Delegated{cls.__name__}", (cls,), delegating_methods
            )

            # Swap the instance's class to our new dynamic class.
            # This is the key step that makes the delegation work transparently.
            # From now on, any method call on `instance` will first check
            # `DynamicDelegator`, then `cls`, and so on.
            instance.__class__ = DynamicDelegator
            return instance
        else:
            # --- Case 1: Direct Implementation Pattern ---
            # No delegation target was found. Enforce standard ABC rules.
            # Check if any abstract methods are left unimplemented.
            missing_methods = [
                name for name in original_abstract_methods
                if not hasattr(instance, name) or getattr(getattr(instance, name), "__isabstractmethod__", False)
            ]
            if missing_methods:
                raise TypeError(
                    f"Cannot instantiate abstract class {cls.__name__} without an "
                    f"implementation for abstract methods: {', '.join(missing_methods)}"
                )
            return instance


class Interface(metaclass=InterfaceMeta):
    """A base class for all components in the framework.

    By inheriting from this class, components automatically gain the flexible
    implementation capabilities provided by the `InterfaceMeta` metaclass.
    This allows them to be implemented either by directly overriding abstract
    methods or by delegating to another object.
    """
    pass
