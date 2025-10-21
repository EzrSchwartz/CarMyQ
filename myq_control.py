"""
MyQ Garage Door Controller
Provides functionality to connect to and control MyQ garage doors
"""

import asyncio
from typing import Optional, List, Dict
from pymyq import login
from pymyq.errors import MyQError, RequestError


class MyQController:
    """Controller for MyQ garage door operations"""

    def __init__(self, email: str, password: str):
        """
        Initialize MyQ controller

        Args:
            email: MyQ account email
            password: MyQ account password
        """
        self.email = email
        self.password = password
        self.myq = None
        self.devices = {}

    async def connect(self) -> bool:
        """
        Connect to MyQ account

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.myq = await login(self.email, self.password)
            self.devices = await self.myq.get_devices()
            print(f"Connected to MyQ. Found {len(self.devices)} device(s)")
            return True
        except MyQError as e:
            print(f"MyQ error during connection: {e}")
            return False
        except RequestError as e:
            print(f"Request error during connection: {e}")
            return False

    async def disconnect(self):
        """Close MyQ connection"""
        if self.myq:
            await self.myq.close()
            print("Disconnected from MyQ")

    def get_garage_doors(self) -> List:
        """
        Get all garage door devices

        Returns:
            List of garage door devices
        """
        garages = []
        for device in self.devices.values():
            if device.device_type == "garagedoor":
                garages.append(device)
        return garages

    async def get_door_status(self, door_name: Optional[str] = None) -> Dict[str, str]:
        """
        Get status of garage door(s)

        Args:
            door_name: Optional specific door name. If None, returns all doors

        Returns:
            Dictionary of door names and their states
        """
        garages = self.get_garage_doors()
        status = {}

        for garage in garages:
            await garage.update()
            if door_name is None or garage.name == door_name:
                status[garage.name] = garage.state

        return status

    async def open_door(self, door_name: Optional[str] = None) -> bool:
        """
        Open garage door(s)

        Args:
            door_name: Optional specific door name. If None, opens all doors

        Returns:
            True if successful, False otherwise
        """
        garages = self.get_garage_doors()

        if not garages:
            print("No garage doors found")
            return False

        success = True
        for garage in garages:
            if door_name is None or garage.name == door_name:
                await garage.update()
                if garage.state == "closed":
                    print(f"Opening {garage.name}...")
                    try:
                        await garage.open()
                    except Exception as e:
                        print(f"Error opening {garage.name}: {e}")
                        success = False
                else:
                    print(f"{garage.name} is already {garage.state}")

        return success

    async def close_door(self, door_name: Optional[str] = None) -> bool:
        """
        Close garage door(s)

        Args:
            door_name: Optional specific door name. If None, closes all doors

        Returns:
            True if successful, False otherwise
        """
        garages = self.get_garage_doors()

        if not garages:
            print("No garage doors found")
            return False

        success = True
        for garage in garages:
            if door_name is None or garage.name == door_name:
                await garage.update()
                if garage.state == "open":
                    print(f"Closing {garage.name}...")
                    try:
                        await garage.close()
                    except Exception as e:
                        print(f"Error closing {garage.name}: {e}")
                        success = False
                else:
                    print(f"{garage.name} is already {garage.state}")

        return success

    async def toggle_door(self, door_name: Optional[str] = None) -> bool:
        """
        Toggle garage door(s) - open if closed, close if open

        Args:
            door_name: Optional specific door name. If None, toggles all doors

        Returns:
            True if successful, False otherwise
        """
        garages = self.get_garage_doors()

        if not garages:
            print("No garage doors found")
            return False

        success = True
        for garage in garages:
            if door_name is None or garage.name == door_name:
                await garage.update()
                if garage.state == "closed":
                    print(f"Opening {garage.name}...")
                    try:
                        await garage.open()
                    except Exception as e:
                        print(f"Error opening {garage.name}: {e}")
                        success = False
                elif garage.state == "open":
                    print(f"Closing {garage.name}...")
                    try:
                        await garage.close()
                    except Exception as e:
                        print(f"Error closing {garage.name}: {e}")
                        success = False
                else:
                    print(f"{garage.name} is currently {garage.state}, waiting...")

        return success

    def list_doors(self) -> List[str]:
        """
        Get list of all garage door names

        Returns:
            List of garage door names
        """
        garages = self.get_garage_doors()
        return [garage.name for garage in garages]


async def main():
    """Example usage of MyQController"""
    # Replace with your MyQ credentials
    import os
    EMAIL = os.environ.get("MYQ_EMAIL", "example")
    PASSWORD = os.environ.get("MYQ_PASSWORD")
    if not PASSWORD:
        print("Error: MYQ_PASSWORD environment variable not set.")
        return

    controller = MyQController(EMAIL, PASSWORD)

    # Connect to MyQ
    if not await controller.connect():
        print("Failed to connect to MyQ")
        return

    try:
        # List all garage doors
        doors = controller.list_doors()
        print(f"\nFound garage doors: {doors}")

        # Get status of all doors
        status = await controller.get_door_status()
        print(f"\nCurrent status: {status}")

        # Toggle the first door (or specify by name)
        # await controller.toggle_door()  # Toggles all doors
        # await controller.toggle_door("Garage Door")  # Toggles specific door

        # Or use specific commands:
        # await controller.open_door()
        # await controller.close_door()

    finally:
        # Always disconnect
        await controller.disconnect()


if __name__ == "__main__":
    asyncio.run(main())