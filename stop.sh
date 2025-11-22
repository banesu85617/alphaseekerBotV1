#!/bin/bash
# AlphaSeeker 停止脚本
# ====================

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

# 停止AlphaSeeker系统
stop_alphaseeker() {
    local force=${1:-false}
    
    log_info "正在停止AlphaSeeker系统..."
    
    # 检查PID文件
    if [ -f "alphaseeker.pid" ]; then
        local pid=$(cat alphaseeker.pid)
        
        if kill -0 $pid 2>/dev/null; then
            log_info "发送TERM信号到进程 $pid..."
            kill -TERM $pid
            
            # 等待进程正常退出
            local count=0
            while kill -0 $pid 2>/dev/null && [ $count -lt 10 ]; do
                sleep 1
                count=$((count + 1))
            done
            
            # 检查是否仍在运行
            if kill -0 $pid 2>/dev/null; then
                if [ "$force" = "true" ]; then
                    log_warning "正常停止失败，使用强制终止..."
                    kill -KILL $pid
                    sleep 1
                else
                    log_error "进程未在10秒内停止"
                    log_info "使用 --force 选项强制终止"
                    return 1
                fi
            fi
            
            log_success "AlphaSeeker已停止 (PID: $pid)"
        else
            log_warning "PID文件存在但进程 $pid 未运行"
        fi
        
        # 清理PID文件
        rm -f alphaseeker.pid
    else
        # 尝试通过进程名查找
        local pids=$(pgrep -f "main_integration.py" || true)
        
        if [ -n "$pids" ]; then
            log_info "找到AlphaSeeker进程: $pids"
            
            if [ "$force" = "true" ]; then
                log_info "强制终止进程..."
                kill -KILL $pids 2>/dev/null || true
            else
                log_info "发送TERM信号到进程..."
                kill -TERM $pids 2>/dev/null || true
                sleep 2
                
                # 检查是否仍在运行
                local remaining_pids=$(pgrep -f "main_integration.py" || true)
                if [ -n "$remaining_pids" ]; then
                    log_warning "部分进程仍在运行，发送KILL信号..."
                    kill -KILL $remaining_pids 2>/dev/null || true
                fi
            fi
            
            log_success "AlphaSeeker进程已终止"
        else
            log_warning "未找到运行中的AlphaSeeker进程"
        fi
    fi
    
    # 检查端口是否释放
    local port=${ALPHASEEKER_PORT:-8000}
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        local port_pid=$(lsof -Pi :$port -sTCP:LISTEN -t)
        log_warning "端口 $port 仍被进程 $port_pid 占用"
        
        if [ "$force" = "true" ]; then
            log_info "强制终止端口占用进程..."
            kill -9 $port_pid 2>/dev/null || true
            sleep 1
            
            if ! lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
                log_success "端口 $port 已释放"
            else
                log_error "无法释放端口 $port"
                return 1
            fi
        else
            log_info "使用 --force 选项释放端口"
            return 1
        fi
    else
        log_success "端口 $port 已释放"
    fi
}

# 清理临时文件
cleanup_temp_files() {
    log_info "清理临时文件..."
    
    local temp_files=(
        "alphaseeker.pid"
        "*.pyc"
        "__pycache__/*"
        ".coverage"
        "*.log.*"
    )
    
    for pattern in "${temp_files[@]}"; do
        if ls $pattern 2>/dev/null; then
            rm -rf $pattern
            log_info "已清理: $pattern"
        fi
    done
}

# 清理日志文件（可选）
cleanup_logs() {
    if [ "$CLEANUP_LOGS" = "true" ]; then
        log_info "清理日志文件..."
        
        if [ -d "logs" ]; then
            # 保留最近5个日志文件
            ls -t logs/alphaseeker.log* 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null || true
            log_info "已清理旧日志文件"
        fi
    fi
}

# 检查系统资源
check_resources() {
    log_info "检查系统资源状态..."
    
    # 检查CPU使用率
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    log_info "CPU使用率: ${cpu_usage}%"
    
    # 检查内存使用率
    local memory_usage=$(free | grep Mem | awk '{printf("%.1f"), $3/$2 * 100.0}')
    log_info "内存使用率: ${memory_usage}%"
    
    # 检查磁盘使用率
    local disk_usage=$(df -h . | tail -1 | awk '{print $5}' | sed 's/%//')
    log_info "磁盘使用率: ${disk_usage}%"
    
    # 检查进程数
    local process_count=$(ps aux | grep -v grep | wc -l)
    log_info "当前进程数: $process_count"
}

# 显示帮助信息
show_help() {
    echo "AlphaSeeker 停止脚本"
    echo ""
    echo "用法: $0 [选项] [命令]"
    echo ""
    echo "命令:"
    echo "  stop       停止系统（默认）"
    echo "  status     检查系统状态"
    echo "  cleanup    清理临时文件"
    echo "  help       显示此帮助信息"
    echo ""
    echo "选项:"
    echo "  --force      强制停止（使用KILL信号）"
    echo "  --port PORT  指定端口号（默认: 8000）"
    echo "  --no-wait    不等待进程正常退出"
    echo ""
    echo "环境变量:"
    echo "  ALPHASEEKER_PORT  指定端口号"
    echo "  CLEANUP_LOGS      清理旧日志文件"
    echo ""
    echo "示例:"
    echo "  $0 stop                # 正常停止"
    echo "  $0 stop --force        # 强制停止"
    echo "  $0 stop --port 8080    # 停止指定端口的服务"
    echo "  $0 cleanup             # 清理临时文件"
}

# 检查系统状态
check_status() {
    log_info "检查AlphaSeeker系统状态..."
    
    local running=false
    
    # 检查PID文件
    if [ -f "alphaseeker.pid" ]; then
        local pid=$(cat alphaseeker.pid)
        if kill -0 $pid 2>/dev/null; then
            log_success "AlphaSeeker正在运行 (PID: $pid)"
            running=true
        else
            log_warning "PID文件存在但进程未运行"
        fi
    fi
    
    # 检查进程
    local pids=$(pgrep -f "main_integration.py" || true)
    if [ -n "$pids" ]; then
        log_success "发现AlphaSeeker进程: $pids"
        running=true
    fi
    
    # 检查端口
    local port=${ALPHASEEKER_PORT:-8000}
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        local port_pid=$(lsof -Pi :$port -sTCP:LISTEN -t)
        log_success "端口 $port 正在监听 (PID: $port_pid)"
        
        if [ "$running" = "false" ]; then
            log_warning "端口监听但未找到相关进程"
        fi
    else
        log_warning "端口 $port 未在监听"
    fi
    
    if [ "$running" = "false" ]; then
        log_warning "AlphaSeeker系统未运行"
    fi
}

# 主函数
main() {
    local command=${1:-stop}
    
    # 解析选项
    shift || true
    local force=false
    local wait=true
    
    while [ $# -gt 0 ]; do
        case $1 in
            --force)
                force=true
                shift
                ;;
            --port)
                shift
                if [ -n "$1" ]; then
                    export ALPHASEEKER_PORT=$1
                fi
                shift
                ;;
            --no-wait)
                wait=false
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
        stop)
            if stop_alphaseeker $force; then
                check_resources
                cleanup_temp_files
                cleanup_logs
                log_success "AlphaSeeker系统已完全停止"
            else
                log_error "停止AlphaSeeker系统失败"
                exit 1
            fi
            ;;
        status)
            check_status
            ;;
        cleanup)
            cleanup_temp_files
            cleanup_logs
            log_success "清理完成"
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