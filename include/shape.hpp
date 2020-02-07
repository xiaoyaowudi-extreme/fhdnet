#include <vector>
#include <cuda.hpp>
#include <stddef.h>
#include <math.h>
#include <vector>
#include <assert.h>

#ifndef _SHAPE_HPP_
#define _SHAPE_HPP_

/**
* +----------------------------------------+
* | @author Ziyao Xiao                     |
* | @array shape                           |
* +----------------------------------------+
*/

namespace fhdnet
{
    class shape
    {
    private:
        std::vector<long long> __shape;
    public:
        shape();

        template<typename T> shape(const std::vector<T> &__in);

        template<typename T, typename IT> shape(T __in[], IT __length);
        
        template<typename T> void assign(const std::vector<T> &__in);

        template<typename T, typename IT> void assign(T __in[], IT __length);

        long long& operator[](const long long &a);

        ~shape();

        struct iterator
        {
            std::vector<long long>::iterator __it;

            void operator++();

            void operator--();

            long long& operator*();

            bool operator!=(const iterator &it);
        };

        struct reverse_iterator
        {
            std::vector<long long>::reverse_iterator __it;

            void operator++();

            void operator--();

            long long& operator*();

            bool operator!=(const reverse_iterator &it);
        };

        iterator begin();

        reverse_iterator rbegin();

        iterator end();

        reverse_iterator rend();

        long long size();
    };
}

namespace fhdnet
{
    template<typename T>
    shape::shape(const std::vector <T> &__in)
    {
        __shape.assign(__in.begin(), __in.end());
    }
    template<typename T, typename IT>
    shape::shape(T __in[], IT __length)
    {
        for(int i = 0; i < __length; ++i)
        {
            __shape.push_back(__in[i]);
        }
    }
    shape::shape(){}
    template<typename T>
    void shape::assign(const std::vector<T> &__in)
    {
        __shape.assign(__in.begin(), __in.end());
    }
    template<typename T, typename IT>
    void shape::assign(T __in[], IT __length)
    {
        while(__shape.size())
        {
            __shape.pop_back();
        }
        for(int i = 0; i < __length; ++i)
        {
            __shape.push_back(__in[i]);
        }
    }
    long long& shape::operator[](const long long &a)
    {
        if(a < 0)
        {
            if((long long)(__shape.size()) + a < 0LL)
            {
                throw "Shape index out of range.";
            }
            return __shape[(long long)(__shape.size()) + a];
        }
        else
        {
            if(a >= __shape.size())
            {
                throw "Shape index out of range.";
            }
            return __shape[a];
        }
    }
    shape::~shape(){}
    void shape::iterator::operator++()
    {
        __it++;
    }
    void shape::iterator::operator--()
    {
        __it--;
    }
    long long& shape::iterator::operator*()
    {
        return *__it;
    }
    bool shape::iterator::operator!=(const shape::iterator &it)
    {
        return __it != it.__it;
    }
    void shape::reverse_iterator::operator++()
    {
        __it++;
    }
    void shape::reverse_iterator::operator--()
    {
        __it--;
    }
    long long& shape::reverse_iterator::operator*()
    {
        return *__it;
    }
    bool shape::reverse_iterator::operator!=(const shape::reverse_iterator &it)
    {
        return __it != it.__it;
    }
    shape::iterator shape::begin()
    {
        return (shape::iterator){__shape.begin()};
    }
    shape::iterator shape::end()
    {
        return (shape::iterator){__shape.end()};
    }
    shape::reverse_iterator shape::rbegin()
    {
        return (shape::reverse_iterator){__shape.rbegin()};
    }
    shape::reverse_iterator shape::rend()
    {
        return (shape::reverse_iterator){__shape.rend()};
    }
    long long shape::size()
    {
        long long __total = 1;
        for(shape::iterator it = this->begin(); it != this->end(); ++it)
        {
            __total *= *it;
            assert((*it) >= 0);
        }
        return __total;
    }
}

#endif