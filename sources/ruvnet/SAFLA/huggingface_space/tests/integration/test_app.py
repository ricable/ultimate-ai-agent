"""
Integration tests for the main SAFLA application.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import gradio as gr

# Import after sys.path manipulation in conftest
from app import SAFLAApp, main
from src.config.settings import AppSettings


@pytest.mark.integration
class TestSAFLAApp:
    """Integration tests for SAFLA application."""
    
    def test_app_initialization(self, test_settings):
        """Test application initialization."""
        with patch('app.AppSettings', return_value=test_settings):
            app = SAFLAApp()
            
            assert app.settings == test_settings
            assert app.safla_manager is not None
            assert app.interface is None  # Not created until create_interface
    
    def test_create_interface(self, test_settings):
        """Test interface creation."""
        with patch('app.AppSettings', return_value=test_settings):
            app = SAFLAApp()
            
            # Mock the async initialization
            with patch.object(app, '_async_init', new_callable=AsyncMock):
                interface = app.create_interface()
                
                assert isinstance(interface, gr.Blocks)
                assert app.interface is not None
    
    @pytest.mark.asyncio
    async def test_get_system_status_healthy(self, test_settings):
        """Test getting healthy system status."""
        with patch('app.AppSettings', return_value=test_settings):
            app = SAFLAApp()
            
            # Mock system status
            app.safla_manager = MagicMock()
            app.safla_manager.get_system_status = AsyncMock(return_value={
                "status": "healthy",
                "memory_initialized": True,
                "safety_active": True,
                "metacognitive_running": True
            })
            
            status = await app._get_system_status()
            
            assert "🟢" in status
            assert "Operational" in status
            assert "Memory: ✅" in status
            assert "Safety: ✅" in status
    
    @pytest.mark.asyncio
    async def test_get_system_status_error(self, test_settings):
        """Test getting system status with error."""
        with patch('app.AppSettings', return_value=test_settings):
            app = SAFLAApp()
            
            # Mock system status to raise error
            app.safla_manager = MagicMock()
            app.safla_manager.get_system_status = AsyncMock(
                side_effect=Exception("Test error")
            )
            
            status = await app._get_system_status()
            
            assert "🔴" in status
            assert "Error retrieving status" in status
    
    def test_get_current_config(self, test_settings):
        """Test getting current configuration."""
        with patch('app.AppSettings', return_value=test_settings):
            app = SAFLAApp()
            
            config = app._get_current_config()
            
            assert "environment" in config
            assert "safla" in config
            assert "performance" in config
            assert "debug" in config
            assert config["environment"] == test_settings.environment
    
    def test_launch_configuration(self, test_settings):
        """Test launch configuration."""
        with patch('app.AppSettings', return_value=test_settings):
            app = SAFLAApp()
            
            # Mock interface
            app.interface = MagicMock()
            app.interface.launch = MagicMock()
            
            # Test default launch
            app.launch()
            
            app.interface.launch.assert_called_once()
            call_args = app.interface.launch.call_args[1]
            
            assert call_args["server_name"] == "0.0.0.0"
            assert call_args["server_port"] == 7860
            assert call_args["share"] is False
            assert call_args["debug"] == test_settings.is_debug()
    
    def test_launch_with_custom_config(self, test_settings):
        """Test launch with custom configuration."""
        with patch('app.AppSettings', return_value=test_settings):
            app = SAFLAApp()
            
            # Mock interface
            app.interface = MagicMock()
            app.interface.launch = MagicMock()
            
            # Test custom launch
            app.launch(server_port=8080, share=True)
            
            app.interface.launch.assert_called_once()
            call_args = app.interface.launch.call_args[1]
            
            assert call_args["server_port"] == 8080
            assert call_args["share"] is True
    
    def test_main_function(self):
        """Test main entry point."""
        with patch('app.SAFLAApp') as mock_app_class:
            mock_app = MagicMock()
            mock_app_class.return_value = mock_app
            
            # Test normal execution
            main()
            
            mock_app_class.assert_called_once()
            mock_app.launch.assert_called_once()
    
    def test_main_keyboard_interrupt(self):
        """Test main with keyboard interrupt."""
        with patch('app.SAFLAApp') as mock_app_class:
            mock_app = MagicMock()
            mock_app.launch.side_effect = KeyboardInterrupt()
            mock_app_class.return_value = mock_app
            
            # Should not raise exception
            main()
    
    def test_component_initialization_error(self, test_settings):
        """Test component initialization with error."""
        with patch('app.AppSettings', return_value=test_settings), \
             patch('app.SAFLAManager', side_effect=Exception("Init error")):
            
            # Should not raise, but use test mode
            app = SAFLAApp()
            
            assert app.safla_manager is not None
            assert app.safla_manager.test_mode is True


@pytest.mark.integration
class TestAppIntegration:
    """Full integration tests for the application."""
    
    def test_full_app_creation(self):
        """Test full application creation and interface building."""
        # This test verifies the entire app can be created without errors
        app = SAFLAApp()
        
        with patch.object(app.safla_manager, 'initialize_system', new_callable=AsyncMock):
            interface = app.create_interface()
            
            assert interface is not None
            assert hasattr(interface, 'launch')
            
            # Check that main components exist
            # This is a smoke test to ensure basic functionality
            assert app.safla_manager is not None
            assert app.settings is not None