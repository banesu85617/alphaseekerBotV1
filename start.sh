#!/bin/bash
# AlphaSeeker 快速启动脚本
# ========================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Python版本
check_python() {
    log_info "检查Python版本..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        log_error "需要Python 3.8或更高版本，当前版本: $PYTHON_VERSION"
        exit 1
    fi
    
    log_success "Python版本检查通过: $PYTHON_VERSION"
}

# 检查依赖
check_dependencies() {
    log_info "检查必需依赖..."
    
    local missing_deps=()
    
    # 检查pip包
    local required_packages=("fastapi" "uvicorn" "lightgbm" "pandas" "numpy" "aiohttp")
    
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import ${package//-/_}" 2>/dev/null; then
            missing_deps+=("$package")
        fi
    done
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_warning "缺少依赖包: ${missing_deps[*]}"
        log_info "正在安装依赖包..."
        
        python3 -m pip install --upgrade pip
        python3 -m pip install fastapi uvicorn lightgbm pandas numpy scipy joblib aiohttp httpx pyyaml
        
        log_success "依赖包安装完成"
    else
        log_success "所有依赖包已安装"
    fi
}

# 创建目录结构
create_directories() {
    log_info "创建目录结构..."
    
    local dirs=("data" "models" "logs" "cache" "config")
    
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_success "创建目录: $dir"
        fi
    done
    
    # 创建子目录
    mkdir -p data/market_data data/training data/backtest
    mkdir -p models/lightgbm models/features models/risk
    log_success "目录结构创建完成"
}

# 检查配置文件
check_config() {
    log_info "检查配置文件..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            log_success "已复制环境变量配置文件: .env"
            log_warning "请编辑 .env 文件以配置您的参数"
        else
            log_warning ".env 文件不存在，将使用默认配置"
        fi
    else
        log_success "环境变量配置文件存在: .env"
    fi
    
    if [ ! -f "config/main_config.yaml" ]; then
        log_warning "主配置文件不存在: config/main_config.yaml"
        log_info "将使用代码中定义的默认配置"
    else
        log_success "主配置文件存在: config/main_config.yaml"
    fi
}

# 检查端口占用
check_port() {
    local port=${1:-8000}
    
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_warning "端口 $port 已被占用"
        
        # 获取占用端口的进程信息
        local pid=$(lsof -Pi :$port -sTCP:LISTEN -t)
        local process=$(ps -p $pid -o comm= 2>/dev/null || echo "未知")
        
        log_info "占用进程: $process (PID: $pid)"
        log_info "您可以:"
        echo "  1. 停止现有进程: kill -9 $pid"
        echo "  2. 使用其他端口: ALPHASEEKER_PORT=$((port+1)) ./start.sh"
        echo "  3. 强制启动（将终止占用进程）: ./start.sh --force"
        
        if [ "$1" = "--force" ]; then
            log_warning "强制模式: 终止占用进程 $pid"
            kill -9 $pid 2>/dev/null || true
            sleep 2
        else
            return 1
        fi
    fi
    
    return 0
}

# 启动系统
start_system() {
    log_info "启动AlphaSeeker系统..."
    
    # 设置环境变量
    export PYTHONPATH="$(pwd)"
    
    # 检查调试模式
    if [ "$DEBUG" = "true" ]; then
        export ALPHASEEKER_DEBUG=true
        export ALPHASEEKER_LOG_LEVEL=DEBUG
        log_info "调试模式已启用"
    fi
    
    # 启动命令
    local cmd="python3 main_integration.py"
    
    if [ "$BACKGROUND" = "true" ]; then
        log_info "后台模式启动..."
        nohup $cmd > logs/alphaseeker.log 2>&1 &
        local pid=$!
        echo $pid > alphaseeker.pid
        log_success "AlphaSeeker已在后台启动 (PID: $pid)"
        log_info "日志文件: logs/alphaseeker.log"
        log_info "停止命令: kill $pid 或 ./stop.sh"
    else
        log_info "前台模式启动..."
        exec $cmd
    fi
}

# 停止系统
stop_system() {
    log_info "停止AlphaSeeker系统..."
    
    if [ -f "alphaseeker.pid" ]; then
        local pid=$(cat alphaseeker.pid)
        if kill -0 $pid 2>/dev/null; then
            kill $pid
            sleep 2
            
            # 强制终止
            if kill -0 $pid 2>/dev/null; then
                log_warning "正常停止失败，使用强制终止"
                kill -9 $pid
            fi
            
            rm -f alphaseeker.pid
            log_success "AlphaSeeker已停止"
        else
            log_warning "进程 $pid 不存在"
            rm -f alphaseeker.pid
        fi
    else
        # 尝试通过进程名查找
        local pids=$(pgrep -f "main_integration.py" || true)
        if [ -n "$pids" ]; then
            log_info "找到AlphaSeeker进程: $pids"
            kill $pids 2>/dev/null || true
            sleep 1
            
            # 强制终止
            pids=$(pgrep -f "main_integration.py" || true)
            if [ -n "$pids" ]; then
                kill -9 $pids 2>/dev/null || true
            fi
            
            log_success "AlphaSeeker已停止"
        else
            log_warning "未找到运行中的AlphaSeeker进程"
        fi
    fi
}

# 检查系统状态
check_status() {
    log_info "检查AlphaSeeker系统状态..."
    
    if [ -f "alphaseeker.pid" ]; then
        local pid=$(cat alphaseeker.pid)
        if kill -0 $pid 2>/dev/null; then
            log_success "AlphaSeeker正在运行 (PID: $pid)"
            
            # 检查端口
            if lsof -Pi :${ALPHASEEKER_PORT:-8000} -sTCP:LISTEN -t >/dev/null 2>&1; then
                log_success "服务端口正常监听"
                
                # 测试API
                if command -v curl &> /dev/null; then
                    if curl -s http://localhost:${ALPHASEEKER_PORT:-8000}/health >/dev/null 2>&1; then
                        log_success "API健康检查通过"
                    else
                        log_warning "API健康检查失败"
                    fi
                fi
            else
                log_warning "服务端口未监听"
            fi
        else
            log_warning "PID文件存在但进程未运行"
            rm -f alphaseeker.pid
        fi
    else
        log_warning "AlphaSeeker未运行"
    fi
}

# 显示帮助信息
show_help() {
    echo "AlphaSeeker 启动脚本"
    echo ""
    echo "用法: $0 [选项] [命令]"
    echo ""
    echo "命令:"
    echo "  start    启动系统（默认）"
    echo "  stop     停止系统"
    echo "  restart  重启系统"
    echo "  status   检查系统状态"
    echo "  demo     运行演示程序"
    echo "  help     显示此帮助信息"
    echo ""
    echo "选项:"
    echo "  --debug      启用调试模式"
    echo "  --background 后台运行"
    echo "  --force      强制启动（终止占用端口的进程）"
    echo "  --port PORT  指定端口号（默认: 8000）"
    echo ""
    echo "环境变量:"
    echo "  DEBUG=true           启用调试模式"
    echo "  BACKGROUND=true      后台运行"
    echo "  ALPHASEEKER_PORT     指定端口号"
    echo ""
    echo "示例:"
    echo "  $0 start                    # 正常启动"
    echo "  $0 start --debug           # 调试模式启动"
    echo "  $0 start --background      # 后台启动"
    echo "  $0 start --port 8080       # 指定端口启动"
    echo "  $0 start --force           # 强制启动"
    echo "  $0 demo                    # 运行演示"
}

# 运行演示
run_demo() {
    log_info "启动演示程序..."
    
    if [ ! -f "demo_complete.py" ]; then
        log_error "演示程序不存在: demo_complete.py"
        exit 1
    fi
    
    # 确保系统正在运行
    if ! curl -s http://localhost:${ALPHASEEKER_PORT:-8000}/health >/dev/null 2>&1; then
        log_warning "AlphaSeeker系统未运行，正在启动..."
        start_system
        sleep 5
    fi
    
    python3 demo_complete.py
}

# 主函数
main() {
    local command=${1:-start}
    
    # 解析选项
    shift || true
    while [ $# -gt 0 ]; do
        case $1 in
            --debug)
                DEBUG=true
                shift
                ;;
            --background)
                BACKGROUND=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --port)
                shift
                if [ -n "$1" ]; then
                    export ALPHASEEKER_PORT=$1
                fi
                shift
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 设置默认端口
    export ALPHASEEKER_PORT=${ALPHASEEKER_PORT:-8000}
    
    # 根据命令执行相应操作
    case $command in
        start)
            echo "🚀 启动AlphaSeeker系统..."
            check_python
            check_dependencies
            create_directories
            check_config
            
            if ! check_port $ALPHASEEKER_PORT; then
                if [ "$FORCE" = "true" ]; then
                    check_port $ALPHASEEKER_PORT --force
                else
                    exit 1
                fi
            fi
            
            start_system
            ;;
        stop)
            echo "🛑 停止AlphaSeeker系统..."
            stop_system
            ;;
        restart)
            echo "🔄 重启AlphaSeeker系统..."
            stop_system
            sleep 2
            check_dependencies
            start_system
            ;;
        status)
            echo "📊 检查AlphaSeeker系统状态..."
            check_status
            ;;
        demo)
            echo "🎬 运行AlphaSeeker演示..."
            run_demo
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "未知命令: $command"
            show_help
            exit 1
            ;;
    esac
}

# 信号处理
trap 'echo -e "\n👋 脚本被中断"; exit 130' INT TERM

# 运行主函数
main "$@"